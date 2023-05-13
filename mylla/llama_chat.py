import llama_cpp.llama as llama
import os
import sys
import ctypes 
import struct
import signal
import io
import time
import hashlib
import json
import numpy as np
from typing import List
from dataclasses import dataclass
from dataclasses import asdict

mylla_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def llama_default_params() ->  llama.llama_context_params:
    params = llama.llama_context_default_params()
    params.n_parts = -1
    params.f16_kv = True
    params.vocab_only = False
    return params

models = {
    # pre PR-1405
    # '7b': os.path.join(mylla_root, "txtmodels", "llama", "7b", "ggml-model-q4_0_ssd.bin"),
    # '13b': os.path.join(mylla_root, "txtmodels", "llama", "13b", "ggml-model-q4_0_ssd.bin"),
    # '30b': os.path.join(mylla_root, "txtmodels", "llama", "30b", "ggml-model-q4_0_ssd.bin"),
    # '65b': os.path.join(mylla_root, "txtmodels", "llama", "65b", "ggml-model-q4_0_ssd.bin"),
    # 'vicuna7buncen': os.path.join(mylla_root, "txtmodels", "vicuna", "7b", "ggml-vicuna-7b-1.0-uncensored-q4_0.bin"),
    # 'vicuna7b': os.path.join(mylla_root, "txtmodels", "vicuna", "7b", "ggml-vicuna-7b-q4_0.bin"),
    # 'vicuna13bfree': os.path.join(mylla_root, "txtmodels", "vicuna", "13b", "vicuna-13b-free-q4_0.bin"),
    # 'vicuna13b': os.path.join(mylla_root, "txtmodels", "vicuna", "13b", "ggml-vicuna-13b-1.1-q4_0.bin"),
    
    'vicuna7b': os.path.join(mylla_root, "txtmodels", "vicuna", "7b", "ggml-vic7b-uncensored-q4_0.bin"),
    'vicuna13b': os.path.join(mylla_root, "txtmodels", "vicuna", "13b", "ggml-vicuna-13b-free-v230502-q4_0.bin"),
}

@dataclass
class LlamaConfig:
    n_ctx: int = 2048
    n_threads: int = 4
    seed: int = 1337
    model: str = 'vicuna7b'
    logits_all: bool = False
    embedding: bool = True
    def llama_params(self) ->  llama.llama_context_params:
        params = llama_default_params()
        params.seed = self.seed
        params.n_ctx = self.n_ctx
        params.logits_all = self.logits_all
        params.embedding = self.embedding
        return params
    def model_fn(self) -> str:
        global models
        return models[self.model]

class Tokens:
    def __init__(self, ctx:llama.llama_context_p, capacity:int):
        self.ctx = ctx
        self.capacity = capacity
        self.tokens = (llama.llama_token * self.capacity)()
        self.n = 0
    
    def tokenize(self, string:str, add_bos:bool = True):
        self.n = llama.llama_tokenize(self.ctx, string, self.tokens, self.capacity, add_bos)
        if self.n < 0:
            print('Error: llama_tokenize() returned {}'.format(n_tokens))
            sys.exit(1)
    
    def to_str(self, begin=0, end=-1):
        if end<0: end += self.n+1
        return "".join(llama.llama_token_to_str(self.ctx, t) for t in self.tokens[begin:end])
    
    def __str__(self): 
        return self.to_str()
    
    def append(self, token):
        self.tokens[self.n] = token
        self.n += 1

    def erase(self, begin, end=None):
        if end is None: end = begin+1
        remaining = self.n - end
        self.tokens[begin:begin+remaining] = self.tokens[end:end+remaining]
        self.n = begin+remaining

    def extend(self, other, begin=0, end=None):
        if end is None: end = other.n
        count = end - begin
        assert(self.n + count <= self.capacity)
        self.tokens[self.n:self.n+count] = other.tokens[begin:end]
        self.n += count

    def data(self):
        return self.tokens

    def size(self):
        return self.n

    def num_free(self):
        return self.capacity - self.n

    def clear(self):
        self.n = 0

    def read_from(self, file):
        _version = struct.unpack('<Q', file.read(8))[0]
        if _version == 0x1:
            _n = struct.unpack('<Q', file.read(8))[0]
            _capacity = struct.unpack('<Q', file.read(8))[0]
            if _capacity > self.capacity: raise ValueError("not enough capacity")
            self.n = _n
            self.capacity = _capacity
            nbytes = self.capacity*ctypes.sizeof(llama.llama_token)
            data = file.read(nbytes)
            ctypes.memmove(self.tokens, data, nbytes)
        else:
            raise ValueError("invalid version number in file. got '%d'." % _version)

    def write_to(self, file):
        version = 0x1
        file.write(struct.pack('<Q', version))
        file.write(struct.pack('<Q', self.n))
        file.write(struct.pack('<Q', self.capacity))
        file.write(ctypes.string_at(ctypes.addressof(self.tokens), self.capacity*ctypes.sizeof(llama.llama_token)))

class Llama:
    def __init__(self, cfg:LlamaConfig):
        self.cfg = cfg
        self.llama_params = self.cfg.llama_params()
        self.ctx = llama.llama_init_from_file(self.cfg.model_fn(), self.llama_params)
    
    def copy_to_state_buffer(self, buf=None):
        size = llama.llama_get_state_size(self.ctx)
        if buf is None:
            buf = ctypes.create_string_buffer(size)
        else:
            assert(size <= len(buf))
        ptr = ctypes.cast(ctypes.addressof(buf), llama.c_ubyte_p)
        llama.llama_copy_state_data(self.ctx, ptr)
        return buf

    def set_state_from_buffer(self, buf):
        size = llama.llama_get_state_size(self.ctx)
        assert(size <= len(buf))
        ptr = ctypes.cast(ctypes.addressof(buf), llama.c_ubyte_p)
        llama.llama_set_state_data(self.ctx, ptr)

    def write_state(self, file, buf=None):
        if type(file) == str: self.save_state(file)
        buf = self.copy_to_state_buffer(buf)
        version = 0x1
        file.write(struct.pack('<Q', version))
        file.write(struct.pack('<Q', len(buf)))
        file.write(buf)
        return buf

    def read_state(self, file, buf=None):
        if type(file) == str: return self.load_state(file)
        expected_size = llama.llama_get_state_size(self.ctx)
        if buf is None:
            buf = ctypes.create_string_buffer(expected_size)
        else:
            assert(expected_size <= len(buf))
        _version = struct.unpack('<Q', file.read(8))[0]
        if _version == 0x1:
            _size = struct.unpack('<Q', file.read(8))[0]
            if _size != expected_size: raise ValueError("invalid 'size' value in file. expected '%d', got '%d'." % (expected_size, _size))
            i = 0
            n = _size
            read = file.readinto(buf)
            assert(read == n)
            self.set_state_from_buffer(buf)
            return True
        else:
            raise ValueError("invalid version number in file. got '%d'." % _version)

    def save_state(self, fn):
        with open(fn, "wb") as file:
            self.write_state(file)

    def load_state(self, fn):
        if not os.path.exists(fn): return False
        with open(fn, "rb") as file:
            return self.read_state(file)

    def make_tokens(self):
        return Tokens(self.ctx, self.cfg.n_ctx)
    
    def tokenize(self, string, add_space=True):
        if add_space and not string.startswith(' '): string = ' ' + string
        tokens = self.make_tokens()
        tokens.tokenize(string)
        return tokens

    def get_logits(self):
        return llama.llama_get_logits(self.ctx)

@dataclass
class SamplerConfig:
    temp           : float = 0.8  # <= 0.0 disabled
    top_k          : int   = 40   # <= 0 to use vocab size
    top_p          : float = 0.95 # 0.95 # 1.0 = disabled
    tfs_z          : float = 1.00 # 1.0 = disabled
    typical_p      : float = 1.00 # 1.0 = disabled
    repeat_last_n  : int   = 64   # last n tokens to penalize (0 = disable penalty, -1 = context size)
    repeat_penalty : float = 1.3  # 1.0 = disabled
    alpha_presence : float = 0.0  # 0.0 = disabled
    alpha_frequency: float = 0.0  # 0.0 = disabled
    mirostat       : int   = 2    # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    mirostat_tau   : float = 5.00 # target entropy
    mirostat_eta   : float = 0.10 # learning rate
    penalize_nl    : bool  = True # consider newlines as a repeatable token

class Sampler:
    def __init__(self, ctx:llama.llama_context_p, cfg: SamplerConfig):
        self.ctx = ctx
        self.cfg = cfg
        self.n_vocab = llama.llama_n_vocab(self.ctx)
        self.n_ctx   = llama.llama_n_ctx(self.ctx)

        self.candidates_buf   = ctypes.create_string_buffer(ctypes.sizeof(llama.llama_token_data) * self.n_vocab)
        self.candidates       = ctypes.cast(self.candidates_buf, llama.llama_token_data_p)
        self.candidates_p_buf = ctypes.create_string_buffer(ctypes.sizeof(llama.llama_token_data_array) * 1)
        self.candidates_p     = ctypes.cast(self.candidates_p_buf, llama.llama_token_data_array_p)
        self.candidates_p[0]  = llama.llama_token_data_array(self.candidates, self.n_vocab, False)
        assert(ctypes.addressof(self.candidates_p[0].data) == ctypes.addressof(self.candidates_p_buf))
        assert(self.candidates_p[0].size == self.n_vocab)
        assert(self.candidates_p[0].sorted == False)
        self.mirostat_mu_buf  = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_float) * 1)
        self.mirostat_mu      = ctypes.cast(self.mirostat_mu_buf, llama.c_float_p)
        self.mirostat_mu[0] = 2.0 * self.cfg.mirostat_tau
        self.last_tokens = Tokens(ctx, self.n_ctx) # contains last tokens to consider for penalties

    def reset(self):
        self.mirostat_mu[0] = 2.0 * self.cfg.mirostat_tau
        self.last_tokens.clear()

    def sample(self, logits:llama.c_float_p, last_n_tokens:Tokens, n_past:int):
        for token_id in range(self.n_vocab):
            self.candidates[token_id] = llama.llama_token_data(token_id, logits[token_id], 0.0)

        self.candidates_p[0]  = llama.llama_token_data_array(self.candidates, self.n_vocab, False)
        # Apply penalties
        nl_logit = logits[llama.llama_token_nl()]

        last_n_past = min(last_n_tokens.size(), n_past)
        last_n_repeat = min(min(last_n_past, self.cfg.repeat_last_n), self.n_ctx)
        self.last_tokens.clear()
        self.last_tokens.extend(last_n_tokens, last_n_past - last_n_repeat, last_n_past)

        llama.llama_sample_repetition_penalty(
            self.ctx, 
            self.candidates_p,
            self.last_tokens.data(),
            self.last_tokens.size(),
            self.cfg.repeat_penalty)
        llama.llama_sample_frequency_and_presence_penalties(
            self.ctx, 
            self.candidates_p,
            self.last_tokens.data(),
            self.last_tokens.size(), 
            self.cfg.alpha_frequency, 
            self.cfg.alpha_presence)

        if not self.cfg.penalize_nl:
            logits[llama.llama_token_nl()] = nl_logit

        ctx = self.ctx

        if self.cfg.temp <= 0:
            # Greedy sampling
            token = llama.llama_sample_token_greedy(ctx, self.candidates_p)
        else:
            if self.cfg.mirostat == 1:
                mirostat_m = 100
                llama.llama_sample_temperature(ctx, self.candidates_p, self.cfg.temp)
                token = llama.llama_sample_token_mirostat(ctx, self.candidates_p, self.cfg.mirostat_tau, self.cfg.mirostat_eta, mirostat_m, self.mirostat_mu)
            elif self.cfg.mirostat == 2:
                llama.llama_sample_temperature(ctx, self.candidates_p, self.cfg.temp)
                token = llama.llama_sample_token_mirostat_v2(ctx, self.candidates_p, self.cfg.mirostat_tau, self.cfg.mirostat_eta, self.mirostat_mu)
            else:
                # Temperature sampling
                llama.llama_sample_top_k        (ctx, self.candidates_p, self.cfg.top_k, 1)
                llama.llama_sample_tail_free    (ctx, self.candidates_p, self.cfg.tfs_z, 1)
                llama.llama_sample_typical      (ctx, self.candidates_p, self.cfg.typical_p, 1)
                
                llama.llama_sample_top_p        (ctx, self.candidates_p, self.cfg.top_p, 1)
                llama.llama_sample_temperature  (ctx, self.candidates_p, self.cfg.temp)
                token = llama.llama_sample_token(ctx, self.candidates_p)
        return token

@dataclass
class ChatConfig:
    llama_cfg: LlamaConfig = LlamaConfig()
    sampler_cfg: SamplerConfig = SamplerConfig()
    prompt: str = ''
    prompt_fn: str = ''
    n_batch: int = 1
    def load_prompt(self) -> str:
        global mylla_root
        if self.prompt_fn != '':
            with io.open(os.path.join(mylla_root, "prompts", self.prompt_fn),"r", encoding="utf-8", errors="ignore") as f:
                prompt = f.read()
            return prompt
        else:
            return self.prompt

def hash_str(str):
    return hashlib.sha256(str.encode('utf-8')).hexdigest()

class LlamaChat:
    def __init__(self, cfg:ChatConfig):
        self.cfg = cfg
        self.llama = Llama(cfg.llama_cfg)
        self.prompt = self.cfg.load_prompt()
        self.sampler = Sampler(self.llama.ctx, self.cfg.sampler_cfg)
        self.tokens_prompt = self.llama.tokenize(self.prompt)
        self.n_past = 0
        self.n_keep = self.tokens_prompt.n
        self.tokens_eval = self.llama.make_tokens()
        self.tokens_echo = self.llama.make_tokens()
        self.tokens = self.llama.make_tokens()
        self.tokens_input = self.llama.make_tokens()
        self._state_buf = None
        self.reset()

    def reset(self):
        self.tokens_eval.clear()
        self.tokens_echo.clear()
        self.tokens.clear()
        self.tokens_input.clear()
        self.tokens_input.extend(self.tokens_prompt)
        self.n_past = 0

    def feed_input(self, echo=True):
        if self.tokens_input.n == 0: return 0
        n = min(self.tokens_input.n, self.cfg.n_batch)
        n = min(n, self.tokens_eval.num_free()) 
        if echo: n = min(n, self.tokens_echo.num_free())
        if self.tokens.num_free() < n:
            # context swapping 
            dynamic = self.n_past - self.n_keep
            keep_dyn = int(dynamic/2)
            self.n_past = self.n_keep
            assert(self.tokens_eval.n == 0)
            assert(self.tokens_eval.num_free() >= keep_dyn)
            self.tokens_eval.extend(self.tokens, self.tokens.n - keep_dyn)
            self.tokens.erase(self.n_keep, self.tokens.n - keep_dyn)
            n = min(n, self.tokens.num_free())
        if echo:
            self.tokens_echo.extend(self.tokens_input, 0, n)
        self.tokens_eval.extend(self.tokens_input, 0, n)
        self.tokens.extend(self.tokens_input, 0, n)
        self.tokens_input.erase(0, n)
        return n

    def get_dynamic_context_capacity(self):
        return self.llama.cfg.n_ctx - self.n_keep
    
    def get_processed_dynamic_context_length(self):
        return self.n_past - self.n_keep

    def get_dynamic_context_keep_length(self):
        return int(self.get_dynamic_context_capacity() / 2)

    def echo_tokens(self):
        string = str(self.tokens_echo)
        self.tokens_echo.clear()
        print(string, end='', flush=True)

    def process_tokens(self):
        if self.tokens_eval.n == 0: 
            return 0
        r = llama.llama_eval(
            self.llama.ctx, 
            self.tokens_eval.tokens, 
            self.tokens_eval.n, 
            self.n_past, 
            self.cfg.llama_cfg.n_threads
        )
        if r != 0:
            print('Error: llama_eval() returned {}'.format(r))
            exit(1)
        self.n_past += self.tokens_eval.n
        n_processed = self.tokens_eval.n
        self.tokens_eval.clear()
        return n_processed

    def sample(self, ignore=None, insert=True):
        max_tries = 16
        for k in range(max_tries):
            logits = self.llama.get_logits()
            token = self.sampler.sample(logits, self.tokens, self.n_past)
            s = llama.llama_token_to_str(self.llama.ctx, token)
            if ignore is None or s not in ignore:
                break
        if insert:
            self.insert_token(token)
        return token

    def insert_token(self, token):
        self.tokens_input.append(token)

    def get_user_input(self):
        return input()

    def process_input(self, echo=True):
        n_processed = 0
        while self.feed_input(echo) > 0:
            self.echo_tokens()
            n_processed += self.process_tokens()
        return n_processed

    def get_max_logit(self):
        my_logits = llama.llama_get_logits(self.llama.ctx) # POINTER(c_float)
        n_vocab = llama.llama_n_vocab(self.llama.ctx)
        max_k = max(range(n_vocab), key=lambda k:my_logits[k])
        max_logit = my_logits[max_k]
        return max_k, max_logit

    def patch_logits(self, other, gain = 0.5, combinator=None):
        if combinator is None: combinator = lambda x,y: y
        n_vocab = llama.llama_n_vocab(self.llama.ctx)
        my_logits = llama.llama_get_logits(self.llama.ctx) # POINTER(c_float)
        other_logits = llama.llama_get_logits(other.llama.ctx) # POINTER(c_float)
        rgain = 1-gain
        for k in range(n_vocab):
            my_logits[k] = (my_logits[k] * rgain + gain * combinator(my_logits[k], other_logits[k]))
            # my_logits[k] = (my_logits[k] * rgain + gain * max(my_logits[k], other_logits[k]))
            # my_logits[k] = other_logits[k]

    def write_state(self, file):
        if type(file) == str: self.save_state(file)
        version = 0x2
        file.write(struct.pack('<Q', version))
        prompt_bytes = self.prompt.encode('utf-8', errors='ignore')
        file.write(struct.pack('<Q', len(prompt_bytes)))
        file.write(prompt_bytes)
        file.write(struct.pack('<Q', self.n_past))
        file.write(struct.pack('<Q', self.n_keep))
        self.tokens_prompt.write_to(file)
        self.tokens_eval.write_to(file)
        self.tokens_echo.write_to(file)
        self.tokens.write_to(file)
        self.tokens_input.write_to(file)
        self._state_buf = self.llama.write_state(file, self._state_buf)

    def read_state(self, file, echo = True):
        if type(file) == str: return self.load_state(file, echo)
        _version = struct.unpack('<Q', file.read(8))[0]
        if _version == 0x2:
            prompt_bytes_len = struct.unpack('<Q', file.read(8))[0]
            prompt_bytes = file.read(prompt_bytes_len)
            self.prompt = prompt_bytes.decode('utf-8', errors='ignore')
            self.n_past = struct.unpack('<Q', file.read(8))[0]
            self.n_keep = struct.unpack('<Q', file.read(8))[0]
            self.tokens_prompt.read_from(file)
            self.tokens_eval.read_from(file)
            self.tokens_echo.read_from(file)
            self.tokens.read_from(file)
            self.tokens_input.read_from(file)
            if echo: print(str(self.tokens), end='', flush=True)
            self.llama.read_state(file, self._state_buf)
        else:
            raise ValueError("invalid version number in file. got '%d'." % _version)

    def save_state(self, fn=None):
        t0 = time.time()
        if fn is None: fn = self.get_state_filename()
        with open(fn, "wb", buffering=5*1024*1024) as file:
            self.write_state(file)
        t1 = time.time()
        # print("saving state took", t1-t0, "s")

    def load_state(self, fn=None, echo = True):
        t0 = time.time()
        if fn is None: fn = self.get_state_filename()
        if not os.path.exists(fn): return False
        with open(fn, "rb", buffering=5*1024*1024) as file:
            result = self.read_state(file, echo)
        t1 = time.time()
        # print("loading state took", t1-t0, "s")
        return result

    def get_state_filename(self):
        fn = ".".join(map(str,[
            self.cfg.prompt_fn,
            hash_str(self.prompt),
            self.cfg.llama_cfg.model,
            self.cfg.llama_cfg.n_ctx,
            self.cfg.llama_cfg.seed,
            *(['logits_all'] if self.cfg.llama_cfg.logits_all else []),
            *(['embedding'] if self.cfg.llama_cfg.embedding else []),
            "bin",
        ]))
        fn = os.path.join(mylla_root, ".cache", fn)
        return fn


@dataclass
class PatchConfig:
    # represents chats[a].patch_logits(chats[b], w, ops[op])
    op: str = ''
    w: float = 0.5
    a: int = 0
    b: int = -1

@dataclass
class ParallelChatConfig:
    chats: List[ChatConfig]
    patch: List[PatchConfig]
    rev_prompt: List[str]
    main: int = 0 # index of chat to sample from

class PreSwappingLlamaChat:
    def __init__(self, cfg: ChatConfig):
        self.chat = LlamaChat(cfg)
        self.pre_swap = LlamaChat(cfg)
        self.llama = self.chat.llama
    def load_state(self):
        self.chat.load_state()
        self.pre_swap.load_state()
    def process_input(self, echo=True):
        n = self.chat.process_input(echo=echo)
        return n
    def insert_token(self, token):
        self.chat.get
        if self.can_insert_into_pre_swap():
            self.pre_swap.insert_token(token)
        self.chat.insert_token(token)
    def sample(self, ignore=None, insert=True):
        token = self.chat.sample(ignore, insert)
        return token
    def patch_logits(self, other, gain = 0.5, combinator=None):
        self.chat.patch_logits(other, gain, combinator)


class ParallelChat:
    def __init__(self, cfg:ParallelChatConfig, ops:dict=None):
        self.ops = ops if ops is not None else {}
        self.cfg = cfg
        self.chats = [LlamaChat(cfg) for cfg in self.cfg.chats]
        self.transcript_txt = self.get_main().prompt
        
    def init_from_saved(self):
        for k, chat in enumerate(self.chats):
            if k == self.cfg.main: continue # do main last
            chat.load_state()
            if chat.process_input() > 0: chat.save_state()
            print("-")

        self.get_main().load_state()
        if self.get_main().process_input() > 0: self.get_main().save_state()
    
    def get_main(self): return self.chats[self.cfg.main]

    def sample(self, insert=True):
        token = self.get_main().sample(ignore=["\\"],insert=False)
        if insert:
            self.insert_token(token)
        return token

    def insert_text(self, txt:str):
        tkns = self.get_main().llama.tokenize(txt, add_space=False)
        for i in range(1, tkns.n):
            token = tkns.tokens[i]
            self.insert_token(token)

    def insert_token(self, token:int):
        txt = llama.llama_token_to_str(self.get_main().llama.ctx, token)
        self.transcript_txt += txt
        for chat in self.chats:
            chat.insert_token(token)

    def process_input(self, echo_main:bool = True, echo_others:bool = False):
        for k, chat in enumerate(self.chats):
            is_main = (k == self.cfg.main)
            echo = echo_main if is_main else echo_others
            chat.process_input(echo=echo)

    def apply_patch(self):
        for p in self.cfg.patch:
            if p.a < len(self.chats) and p.b < len(self.chats):
                self.chats[p.a].patch_logits(self.chats[p.b], p.w, self.ops.get(p.op, None))
