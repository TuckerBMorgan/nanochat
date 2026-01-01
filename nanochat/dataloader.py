from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


class CurriculumDataLoader:
    """
    Curriculum learning data loader for base pretraining.

    Sorts documents by Flesch Reading Ease score (easy to hard) and
    trains on progressively harder data as loss decreases.

    Usage:
        loader = CurriculumDataLoader(B, T, num_tiers=10, device="cuda")
        for inputs, targets, state_dict in loader:
            loss = train_step(inputs, targets)
            if loss < threshold:
                loader.advance_tier()
    """

    def __init__(
        self,
        B: int,
        T: int,
        split: str = "train",
        num_tiers: int = 10,
        tokenizer_threads: int = 4,
        tokenizer_batch_size: int = 128,
        device: str = "cuda",
        resume_state_dict: dict = None,
    ):
        """
        Initialize curriculum data loader.

        Args:
            B: Batch size
            T: Sequence length
            split: "train" or "val"
            num_tiers: Number of difficulty tiers (default 10 = 10% chunks)
            tokenizer_threads: Number of threads for tokenization
            tokenizer_batch_size: Batch size for tokenization
            device: Device to put tensors on
            resume_state_dict: State dict for resuming (includes current_tier, doc_cursor, sorted_doc_indices)
        """
        assert split in ["train", "val"], "split must be 'train' or 'val'"

        from nanochat.curriculum import flesch_reading_ease

        self.B = B
        self.T = T
        self.split = split
        self.num_tiers = num_tiers
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer_batch_size = tokenizer_batch_size
        self.device = device

        # Get DDP info
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        # Get parquet files
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        self.parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

        # Resume or initialize state
        if resume_state_dict is not None and "sorted_doc_indices" in resume_state_dict:
            # Resuming from checkpoint
            self.current_tier = resume_state_dict["current_tier"]
            self.doc_cursor = resume_state_dict["doc_cursor"]
            self.sorted_doc_indices = resume_state_dict["sorted_doc_indices"]
            self.tier_boundaries = resume_state_dict["tier_boundaries"]
            print(f"[Rank {ddp_rank}] Resuming curriculum at tier {self.current_tier}, cursor {self.doc_cursor}")
        else:
            # Fresh start - score all documents and sort by difficulty
            print(f"[Rank {ddp_rank}] Scoring documents for curriculum learning...")
            doc_scores = []  # List of (global_doc_idx, flesch_score)
            global_doc_idx = 0

            for pq_idx, filepath in enumerate(self.parquet_paths):
                pf = pq.ParquetFile(filepath)
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column('text').to_pylist()
                    for text in texts:
                        score = flesch_reading_ease(text)
                        doc_scores.append((global_doc_idx, pq_idx, rg_idx, score))
                        global_doc_idx += 1

            print(f"[Rank {ddp_rank}] Scored {len(doc_scores)} documents")

            # Sort by Flesch score descending (high score = easy = first)
            doc_scores.sort(key=lambda x: -x[3])

            # Store sorted indices: list of (pq_idx, rg_idx, local_idx_in_rg)
            # We need to reconstruct which document within each row group
            self.sorted_doc_indices = []
            doc_counts = {}  # (pq_idx, rg_idx) -> count seen so far

            # First pass: count docs per row group
            rg_sizes = {}
            for pq_idx, filepath in enumerate(self.parquet_paths):
                pf = pq.ParquetFile(filepath)
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    rg_sizes[(pq_idx, rg_idx)] = len(rg.column('text').to_pylist())

            # Build sorted_doc_indices with full location info
            self.sorted_doc_indices = []
            for global_idx, pq_idx, rg_idx, score in doc_scores:
                self.sorted_doc_indices.append((pq_idx, rg_idx, global_idx, score))

            # Calculate tier boundaries (indices into sorted_doc_indices)
            total_docs = len(self.sorted_doc_indices)
            tier_size = total_docs // num_tiers
            self.tier_boundaries = [i * tier_size for i in range(num_tiers)]
            self.tier_boundaries.append(total_docs)  # End boundary

            self.current_tier = 0
            self.doc_cursor = 0

            print(f"[Rank {ddp_rank}] Curriculum initialized: {total_docs} docs, {num_tiers} tiers")
            print(f"[Rank {ddp_rank}] Tier 0 (easiest): docs 0-{self.tier_boundaries[1]}, "
                  f"Flesch scores {self.sorted_doc_indices[0][3]:.1f} to {self.sorted_doc_indices[self.tier_boundaries[1]-1][3]:.1f}")

        # Initialize tokenizer
        self.tokenizer = get_tokenizer()
        self.bos_token = self.tokenizer.get_bos_token_id()

        # Token buffer
        self.token_buffer = deque()
        self.needed_tokens = B * T + 1

        # Cache for loaded row groups
        self._rg_cache = {}

    @property
    def current_tier_end(self) -> int:
        """Index of last document available in current curriculum (cumulative)."""
        return self.tier_boundaries[self.current_tier + 1]

    def advance_tier(self) -> bool:
        """
        Advance to the next difficulty tier.

        Returns:
            True if advanced, False if already at max tier
        """
        if self.current_tier >= self.num_tiers - 1:
            return False

        self.current_tier += 1
        print(f"[Rank {self.ddp_rank}] Advanced to tier {self.current_tier}/{self.num_tiers - 1}")
        print(f"[Rank {self.ddp_rank}] Now training on docs 0-{self.current_tier_end} "
              f"(Flesch scores {self.sorted_doc_indices[0][3]:.1f} to "
              f"{self.sorted_doc_indices[self.current_tier_end - 1][3]:.1f})")
        return True

    def get_state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "current_tier": self.current_tier,
            "doc_cursor": self.doc_cursor,
            "sorted_doc_indices": self.sorted_doc_indices,
            "tier_boundaries": self.tier_boundaries,
        }

    def _load_document(self, doc_idx: int) -> str:
        """Load a single document by its index in sorted_doc_indices."""
        pq_idx, rg_idx, global_idx, score = self.sorted_doc_indices[doc_idx]

        # Check cache
        cache_key = (pq_idx, rg_idx)
        if cache_key not in self._rg_cache:
            filepath = self.parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            rg = pf.read_row_group(rg_idx)
            self._rg_cache[cache_key] = rg.column('text').to_pylist()
            # Limit cache size
            if len(self._rg_cache) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self._rg_cache))
                del self._rg_cache[oldest_key]

        # Find local index within row group
        # global_idx tells us the absolute position, we need to find position within this rg
        texts = self._rg_cache[cache_key]

        # Calculate local index: count how many docs came before this rg
        docs_before = 0
        for prev_pq in range(pq_idx):
            pf = pq.ParquetFile(self.parquet_paths[prev_pq])
            for prev_rg in range(pf.num_row_groups):
                prev_filepath = self.parquet_paths[prev_pq]
                prev_pf = pq.ParquetFile(prev_filepath)
                prev_rg_data = prev_pf.read_row_group(prev_rg)
                docs_before += len(prev_rg_data.column('text').to_pylist())

        if pq_idx > 0 or rg_idx > 0:
            pf = pq.ParquetFile(self.parquet_paths[pq_idx])
            for prev_rg in range(rg_idx):
                rg_data = pf.read_row_group(prev_rg)
                docs_before += len(rg_data.column('text').to_pylist())

        local_idx = global_idx - docs_before
        return texts[local_idx]

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch of (inputs, targets, state_dict)."""
        use_cuda_optimizations = self.device == "cuda"

        # Accumulate enough tokens
        while len(self.token_buffer) < self.needed_tokens:
            # Get documents from current curriculum (cumulative up to current tier)
            # Distribute across DDP ranks
            batch_texts = []
            for _ in range(self.tokenizer_batch_size):
                # Wrap around within available tiers
                if self.doc_cursor >= self.current_tier_end:
                    self.doc_cursor = 0

                # DDP: each rank processes different documents
                actual_idx = self.doc_cursor + self.ddp_rank
                if actual_idx < self.current_tier_end:
                    text = self._load_document(actual_idx)
                    batch_texts.append(text)

                self.doc_cursor += self.ddp_world_size

            if batch_texts:
                token_lists = self.tokenizer.encode(batch_texts, prepend=self.bos_token, num_threads=self.tokenizer_threads)
                for tokens in token_lists:
                    self.token_buffer.extend(tokens)

        # Build batch
        tokens = [self.token_buffer.popleft() for _ in range(self.needed_tokens)]
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)

        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]

        inputs = inputs_cpu.view(self.B, self.T).to(device=self.device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(self.B, self.T).to(device=self.device, non_blocking=use_cuda_optimizations)

        state_dict = self.get_state_dict()
        return inputs, targets, state_dict


class RandomDataLoader:
    """
    Random baseline data loader for comparing against curriculum learning.

    Uses the same tier structure (number of tiers, tier sizes, advancement logic)
    as CurriculumDataLoader, but fills each tier with randomly selected documents
    instead of ordering by Flesch difficulty. No Flesch scoring is performed.

    This provides a true "random scheduling" baseline to compare against
    "curriculum scheduling" - same structure, completely random content assignment.

    Usage:
        loader = RandomDataLoader(B, T, num_tiers=10, device="cuda", seed=42)
        for inputs, targets, state_dict in loader:
            loss = train_step(inputs, targets)
            if loss < threshold:
                loader.advance_tier()
    """

    def __init__(
        self,
        B: int,
        T: int,
        split: str = "train",
        num_tiers: int = 10,
        tokenizer_threads: int = 4,
        tokenizer_batch_size: int = 128,
        device: str = "cuda",
        resume_state_dict: dict = None,
        seed: int = 42,
    ):
        """
        Initialize random baseline data loader.

        Args:
            B: Batch size
            T: Sequence length
            split: "train" or "val"
            num_tiers: Number of tiers (same structure as curriculum, default 10)
            tokenizer_threads: Number of threads for tokenization
            tokenizer_batch_size: Batch size for tokenization
            device: Device to put tensors on
            resume_state_dict: State dict for resuming
            seed: Random seed for reproducible shuffling
        """
        import random

        assert split in ["train", "val"], "split must be 'train' or 'val'"

        self.B = B
        self.T = T
        self.split = split
        self.num_tiers = num_tiers
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer_batch_size = tokenizer_batch_size
        self.device = device
        self.seed = seed

        # Get DDP info
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        # Get parquet files
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        self.parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

        # Resume or initialize state
        if resume_state_dict is not None and "shuffled_doc_indices" in resume_state_dict:
            # Resuming from checkpoint
            self.current_tier = resume_state_dict["current_tier"]
            self.doc_cursor = resume_state_dict["doc_cursor"]
            self.shuffled_doc_indices = resume_state_dict["shuffled_doc_indices"]
            self.tier_boundaries = resume_state_dict["tier_boundaries"]
            print(f"[Rank {ddp_rank}] Resuming random baseline at tier {self.current_tier}, cursor {self.doc_cursor}")
        else:
            # Fresh start - enumerate all documents (no Flesch scoring)
            print(f"[Rank {ddp_rank}] Enumerating documents for random baseline (no Flesch scoring)...")
            all_docs = []  # List of (pq_idx, rg_idx, global_idx)
            global_doc_idx = 0

            for pq_idx, filepath in enumerate(self.parquet_paths):
                pf = pq.ParquetFile(filepath)
                for rg_idx in range(pf.num_row_groups):
                    # Use metadata to get row count without reading data (much faster)
                    num_docs_in_rg = pf.metadata.row_group(rg_idx).num_rows
                    for _ in range(num_docs_in_rg):
                        all_docs.append((pq_idx, rg_idx, global_doc_idx))
                        global_doc_idx += 1

            total_docs = len(all_docs)
            print(f"[Rank {ddp_rank}] Found {total_docs} documents")

            # Shuffle all documents randomly using seeded RNG
            rng = random.Random(seed)
            rng.shuffle(all_docs)

            # Store shuffled indices (no score needed, use 0.0 as placeholder for compatibility)
            self.shuffled_doc_indices = [(pq_idx, rg_idx, global_idx, 0.0) for pq_idx, rg_idx, global_idx in all_docs]

            # Calculate tier boundaries (same structure as curriculum)
            tier_size = total_docs // num_tiers
            self.tier_boundaries = [i * tier_size for i in range(num_tiers)]
            self.tier_boundaries.append(total_docs)

            self.current_tier = 0
            self.doc_cursor = 0

            print(f"[Rank {ddp_rank}] Random baseline initialized: {total_docs} docs, {num_tiers} tiers (fully random order)")
            print(f"[Rank {ddp_rank}] Tier 0: docs 0-{self.tier_boundaries[1]} (random selection from all data)")

        # Initialize tokenizer
        self.tokenizer = get_tokenizer()
        self.bos_token = self.tokenizer.get_bos_token_id()

        # Token buffer
        self.token_buffer = deque()
        self.needed_tokens = B * T + 1

        # Cache for loaded row groups
        self._rg_cache = {}

    @property
    def current_tier_end(self) -> int:
        """Index of last document available in current tier (cumulative)."""
        return self.tier_boundaries[self.current_tier + 1]

    def advance_tier(self) -> bool:
        """
        Advance to the next tier.

        Returns:
            True if advanced, False if already at max tier
        """
        if self.current_tier >= self.num_tiers - 1:
            return False

        self.current_tier += 1
        print(f"[Rank {self.ddp_rank}] Advanced to tier {self.current_tier}/{self.num_tiers - 1}")
        print(f"[Rank {self.ddp_rank}] Now training on docs 0-{self.current_tier_end} (randomly ordered)")
        return True

    def get_state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "current_tier": self.current_tier,
            "doc_cursor": self.doc_cursor,
            "shuffled_doc_indices": self.shuffled_doc_indices,
            "tier_boundaries": self.tier_boundaries,
        }

    def _load_document(self, doc_idx: int) -> str:
        """Load a single document by its index in shuffled_doc_indices."""
        pq_idx, rg_idx, global_idx, score = self.shuffled_doc_indices[doc_idx]

        # Check cache
        cache_key = (pq_idx, rg_idx)
        if cache_key not in self._rg_cache:
            filepath = self.parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            rg = pf.read_row_group(rg_idx)
            self._rg_cache[cache_key] = rg.column('text').to_pylist()
            # Limit cache size
            if len(self._rg_cache) > 100:
                oldest_key = next(iter(self._rg_cache))
                del self._rg_cache[oldest_key]

        texts = self._rg_cache[cache_key]

        # Calculate local index: count how many docs came before this rg
        docs_before = 0
        for prev_pq in range(pq_idx):
            pf = pq.ParquetFile(self.parquet_paths[prev_pq])
            for prev_rg in range(pf.num_row_groups):
                prev_filepath = self.parquet_paths[prev_pq]
                prev_pf = pq.ParquetFile(prev_filepath)
                prev_rg_data = prev_pf.read_row_group(prev_rg)
                docs_before += len(prev_rg_data.column('text').to_pylist())

        if pq_idx > 0 or rg_idx > 0:
            pf = pq.ParquetFile(self.parquet_paths[pq_idx])
            for prev_rg in range(rg_idx):
                rg_data = pf.read_row_group(prev_rg)
                docs_before += len(rg_data.column('text').to_pylist())

        local_idx = global_idx - docs_before
        return texts[local_idx]

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch of (inputs, targets, state_dict)."""
        use_cuda_optimizations = self.device == "cuda"

        # Accumulate enough tokens
        while len(self.token_buffer) < self.needed_tokens:
            batch_texts = []
            for _ in range(self.tokenizer_batch_size):
                # Wrap around within available tiers
                if self.doc_cursor >= self.current_tier_end:
                    self.doc_cursor = 0

                # DDP: each rank processes different documents
                actual_idx = self.doc_cursor + self.ddp_rank
                if actual_idx < self.current_tier_end:
                    text = self._load_document(actual_idx)
                    batch_texts.append(text)

                self.doc_cursor += self.ddp_world_size

            if batch_texts:
                token_lists = self.tokenizer.encode(batch_texts, prepend=self.bos_token, num_threads=self.tokenizer_threads)
                for tokens in token_lists:
                    self.token_buffer.extend(tokens)

        # Build batch
        tokens = [self.token_buffer.popleft() for _ in range(self.needed_tokens)]
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)

        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]

        inputs = inputs_cpu.view(self.B, self.T).to(device=self.device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(self.B, self.T).to(device=self.device, non_blocking=use_cuda_optimizations)

        state_dict = self.get_state_dict()
        return inputs, targets, state_dict
