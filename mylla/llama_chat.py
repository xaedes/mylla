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
    params.n_ctx = 2048
    params.n_parts = -1
    params.seed = 1337 # 1679473604
    params.f16_kv = True
    params.logits_all = False
    params.vocab_only = False
    return params

models = {
    '7b': os.path.join(mylla_root, "txtmodels", "llama", "7b", "ggml-model-q4_0_ssd.bin"),
    '13b': os.path.join(mylla_root, "txtmodels", "llama", "13b", "ggml-model-q4_0_ssd.bin"),
    '30b': os.path.join(mylla_root, "txtmodels", "llama", "30b", "ggml-model-q4_0_ssd.bin"),
    '65b': os.path.join(mylla_root, "txtmodels", "llama", "65b", "ggml-model-q4_0_ssd.bin")
}

@dataclass
class LlamaConfig:
    n_ctx: int = 2048
    n_threads: int = 4
    seed: int = 1337
    model: str = '7b'
    def llama_params(self) ->  llama.llama_context_params:
        params = llama_default_params()
        params.seed = self.seed
        params.n_ctx = self.n_ctx
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
    
    def kv_cache(self):
        size = llama.llama_get_kv_cache_size(self.ctx)
        ntok = llama.llama_get_kv_cache_token_count(self.ctx)
        dataptr = llama.llama_get_kv_cache(self.ctx)
        voidptr = ctypes.cast(dataptr, ctypes.c_void_p)
        # arr = np.ctypeslib.as_array(voidptr, shape=(size,))
        buf = (ctypes.c_char * size).from_address(voidptr.value)
        return buf, ntok

    def write_kv_cache(self, file):
        if type(file) == str: self.save_kv_cache(file)
        kv, ntok = self.kv_cache()
        version = 0x1
        file.write(struct.pack('<Q', version))
        file.write(struct.pack('<Q', len(kv)))
        file.write(struct.pack('<q', ntok))
        file.write(kv)

    def read_kv_cache(self, file):
        if type(file) == str: return self.load_kv_cache(file)
        kv, ntok = self.kv_cache()
        n = len(kv)
        i = 0
        _version = struct.unpack('<Q', file.read(8))[0]
        if _version == 0x1:
            _n = struct.unpack('<Q', file.read(8))[0]
            _ntok = struct.unpack('<q', file.read(8))[0]
            if _n != n: raise ValueError("invalid 'n' value in file. expected '%d', got '%d'." % (n, _n))
            # if _ntok != ntok: raise ValueError("invalid 'ntok' value in file. expected '%d', got '%d'." % (ntok, _ntok))
            
            read = file.readinto(kv)
            assert(read == n)

            charp = ctypes.addressof(kv)
            voidp = ctypes.cast(charp, ctypes.c_void_p)
            datap = ctypes.cast(voidp, llama.c_ubyte_p)
            llama.llama_set_kv_cache(self.ctx, datap, _n, _ntok)
            return True
        else:
            raise ValueError("invalid version number in file. got '%d'." % _version)

    def save_kv_cache(self, fn):
        with open(fn, "wb") as file:
            self.write_kv_cache(file)

    def load_kv_cache(self, fn):
        if not os.path.exists(fn): return False
        with open(fn, "rb") as file:
            return self.read_kv_cache(file)

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
    
    def tokenize(self, string):
        if not string.startswith(' '): string = ' ' + string
        tokens = self.make_tokens()
        tokens.tokenize(string)
        return tokens

@dataclass
class SamplingConfig:
    top_k:int = 40
    top_p:float = 0.95
    temp:float = 0.8
    repeat_penalty:float = 1.3
    repeat_last_n: int = 64

@dataclass
class ChatConfig:
    llama_cfg: LlamaConfig = LlamaConfig()
    sampling_cfg: SamplingConfig = SamplingConfig()
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
        self.tokens_prompt = self.llama.tokenize(self.prompt)
        self.n_past = 0
        self.n_keep = self.tokens_prompt.n
        self.tokens_eval = self.llama.make_tokens()
        self.tokens_echo = self.llama.make_tokens()
        self.tokens = self.llama.make_tokens()
        self.tokens_last = self.llama.make_tokens()
        self.tokens_input = self.llama.make_tokens()
        self._state_buf = None
        self.reset()

    def reset(self):
        self.tokens_eval.clear()
        self.tokens_echo.clear()
        self.tokens.clear()
        self.tokens_last.clear()
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

    def sample(self, ignore=None):
        self.tokens_last.n = min(self.n_past, self.cfg.sampling_cfg.repeat_last_n)
        self.tokens_last.tokens[:self.tokens_last.n] = self.tokens.tokens[self.n_past-self.tokens_last.n:self.n_past]
        max_tries = 16
        for k in range(max_tries):
            # print("self.tokens_last", ctypes.addressof(self.tokens_last.tokens))
            # print("self.tokens_last.n", self.tokens_last.n)
            token = llama.llama_sample_top_p_top_k(
                self.llama.ctx, 
                self.tokens_last.tokens, 
                self.tokens_last.n, 
                top_k = self.cfg.sampling_cfg.top_k,
                top_p = self.cfg.sampling_cfg.top_p,
                temp = self.cfg.sampling_cfg.temp,
                repeat_penalty = self.cfg.sampling_cfg.repeat_penalty
            )
            s = llama.llama_token_to_str(self.llama.ctx, token)
            if ignore is None or s not in ignore:
                break
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
        version = 0x1
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
        self.tokens_last.write_to(file)
        self.tokens_input.write_to(file)
        self._state_buf = self.llama.write_state(file, self._state_buf)

    def read_state(self, file, echo = True):
        if type(file) == str: return self.load_state(file, echo)
        _version = struct.unpack('<Q', file.read(8))[0]
        if _version == 0x1:
            prompt_bytes_len = struct.unpack('<Q', file.read(8))[0]
            prompt_bytes = file.read(prompt_bytes_len)
            self.prompt = prompt_bytes.decode('utf-8', errors='ignore')
            self.n_past = struct.unpack('<Q', file.read(8))[0]
            self.n_keep = struct.unpack('<Q', file.read(8))[0]
            self.tokens_prompt.read_from(file)
            self.tokens_eval.read_from(file)
            self.tokens_echo.read_from(file)
            self.tokens.read_from(file)
            self.tokens_last.read_from(file)
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
            "bin"
        ]))
        fn = os.path.join(mylla_root, ".cache", fn)
        return fn


@dataclass
class PatchConfig:
    # represents chats[a].patch_logits(chats[b], w, op)
    op: str = ''
    w: float = 0.5
    a: int = 0
    b: int = -1

@dataclass
class ParallelChatConfig:
    chats: List[ChatConfig]
    patch: List[PatchConfig]
    main: int = 0 # index of chat to sample from

class ParallelChat:
    def __init__(self, cfg:ParallelChatConfig, ops:dict=None):
        self.ops = ops if ops is not None else {}
        self.cfg = cfg
        self.chats = [LlamaChat(cfg) for cfg in self.cfg.chats]
        
    def init_from_saved(self):
        for k, chat in enumerate(self.chats):
            if k == self.cfg.main: continue # do main last
            chat.load_state()
            if chat.process_input() > 0: chat.save_state()
            print("-")

        self.get_main().load_state()
        if self.get_main().process_input() > 0: self.get_main().save_state()
    
    def get_main(self): return self.chats[self.cfg.main]

    def sample(self):
        token = self.get_main().sample(ignore=["\\"])
        for k, chat in enumerate(self.chats):
            if k == self.cfg.main: continue 
            chat.insert_token(token)
        return token

    def insert_text(self, txt:str):
        tkns = self.get_main().llama.tokenize(' ' + txt)
        for i in range(1, tkns.n):
            token = tkns.tokens[i]
            for chat in self.chats:
                chat.insert_token(token)

    def process_input(self, echo_main:bool = True, echo_others:bool = False):
        for k, chat in enumerate(self.chats):
            is_main = (k == self.cfg.main)
            echo = echo_main if is_main else echo_others
            chat.process_input(echo=echo)

    def apply_patch(self):
        for p in self.cfg.patch:
            self.chats[p.a].patch_logits(self.chats[p.b], p.w, self.ops.get(p.op, None))
