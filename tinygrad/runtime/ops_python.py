# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Tuple, List, Optional, Any, Dict
import pickle, base64, itertools, time, math
from tinygrad.dtype import DType, dtypes, ImageDType
from tinygrad.helpers import all_same, getenv, flatten
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.codegen.kernel import LinearizerOptions

def exec_alu(arg, dtype, p):
  # TODO: make this complete and correctly honor the dtypes
  # TODO: use this for constant folding
  if arg == TernaryOps.MULACC: return p[0]*p[1]+p[2]
  if arg == TernaryOps.WHERE: return p[1] if p[0] else p[2]
  if arg == UnaryOps.LOG2: return math.log2(p[0]) if p[0] > 0 else math.nan
  if arg == UnaryOps.EXP2: return math.exp(p[0]*math.log(2))
  if arg == UnaryOps.SQRT: return math.sqrt(p[0]) if p[0] > 0 else math.nan
  if arg == UnaryOps.SIN: return math.sin(p[0])
  if arg == UnaryOps.NEG: return -p[0]
  if arg == BinaryOps.MUL: return p[0]*p[1]
  if arg == BinaryOps.ADD: return p[0]+p[1]
  if arg == BinaryOps.SUB: return p[0]-p[1]
  if arg == BinaryOps.XOR: return p[0]^p[1]
  if arg == BinaryOps.MAX: return max(p[0], p[1])
  if arg == BinaryOps.CMPEQ: return p[0] == p[1]
  if arg == BinaryOps.CMPLT: return p[0] < p[1]
  if arg == BinaryOps.DIV: return p[0]//p[1] if dtypes.is_int(dtype) else (p[0]/p[1] if p[1] != 0 else math.nan)
  if arg == BinaryOps.MOD: return p[0]%p[1]
  raise NotImplementedError(f"no support for {arg}")

def _load(m, i):
  if i<0 or i>=len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]
def load(inp, j=0):
  if len(inp) == 4:
    return [_load(m, x+j) if gate else default for m,x,gate,default in zip(*inp)]
  else:
    return [_load(m, x+j) for m,x in zip(inp[0], inp[1])]

def _store(m, i, v):
  if i<0 or i>=len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

class PythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: List[Tuple[UOps, Optional[DType], List[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    start_time = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for op in self.uops:
      print(op)
    print("---")
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      data_list: Dict[int, Any] = {}
      datatype_list: Dict[int, DType] = {}
      pbufs: List[memoryview] = list(bufs)
      i = 0
      loop_ends: Dict[int, int] = {}
      while i < len(self.uops):
        uop, dtype, data_indices, arg = self.uops[i]
        print(i)
        print(self.uops[i])
        print(data_list.keys())
        input = [data_list[v] for v in data_indices]
        datatypes = [datatype_list[v] for v in data_indices]
        if getenv("TRACE"): print(i, uop, dtype, arg, input, datatypes)

        if uop in [UOps.END, UOps.BARRIER, UOps.STORE, UOps.IF]:
          assert dtype is None
          dtype = datatype_list[data_indices[0]]
        assert dtype is not None, f"{uop} is missing a dtype"
        datatype_list[i] = dtype

        if uop is UOps.DEFINE_GLOBAL:
          assert dtype.fmt is not None
          data_list[i] = [pbufs.pop(0).cast(dtype.fmt)] * warp_size
        elif uop is UOps.DEFINE_LOCAL:
          assert dtype.fmt is not None
          lbuf = memoryview(bytearray(arg[1]*dtype.itemsize))
          data_list[i] = [lbuf.cast(dtype.fmt)] * warp_size
        elif uop is UOps.BARRIER:
          # in the python emulator, the warp is always in sync
          data_list[i] = data_list[i - 1]
          i += 1
          continue
        elif uop is UOps.STORE:
          assert len(input) <= 3, "gated stores not supported yet"
          if isinstance(datatypes[0], ImageDType):
            # image store
            assert datatypes[2].sz == 4
            for j, val in enumerate(input[2]):
              for m, ox, oy, v in zip(input[0], input[1][0], input[1][1], val):
                assert ox >= 0 and ox < datatypes[0].shape[1] and oy >= 0 and oy < datatypes[0].shape[0]
                _store(m, ox * 4 + oy * datatypes[0].shape[1] * 4 + j, v)
          elif datatypes[2].sz > 1:
            for j, val in enumerate(input[2]):
              for m, o, v in zip(input[0], input[1], val): _store(m, o + j, v)
          else:
            for m, o, v in zip(*input): _store(m, o, v)
          data_list[i] = m
          i += 1
          continue
        elif uop is UOps.END:
          loop_ends[data_indices[0]] = i
          i = data_indices[0]
          continue
        elif uop is UOps.SPECIAL:
          if arg[1][0] == 'g':
            data_list[i] = [idxs[2-arg[0]]] * warp_size
          elif arg[1][0] == 'l':
            data_list[i] = [x[2-arg[0]] for x in warp]
        elif uop is UOps.CONST: data_list[i] = [int(arg) if dtypes.is_int(dtype) else float(arg)] * warp_size
        elif uop is UOps.DEFINE_ACC:
          if dtype.sz > 1:
            data_list[i] = [[arg] * warp_size for _ in range(dtype.sz)]
          else:
            data_list[i] = [arg] * warp_size
        elif uop is UOps.LOOP:
          if i not in data_list:
            data_list[i] = [0] * warp_size
          else:
            for j in range(len(data_list[i])):
              data_list[i][j] += 1
            if data_list[i][0] == input[1][0]:
              i = loop_ends[i] + 1
              continue
        elif uop is UOps.CAST:
          if dtype.sz > 1:
            data_list[i] = input
          else:
            # TODO: add real cast
            if dtypes.is_int(dtype):
              data_list[i] = [int(x) for x in input[0]]
            elif dtypes.is_float(dtype):
              data_list[i] = [float(x) for x in input[0]]
            else:
              data_list[i] = input[0]
        elif uop is UOps.LOAD:
          if isinstance(datatypes[0], ImageDType):
            assert dtype.sz == 4
            data_list[i] = []
            for j in range(dtype.sz):
              ret = []
              for m,ox,oy in zip(input[0], input[1][0], input[1][1]):
                if ox < 0 or ox >= datatypes[0].shape[1] or oy < 0 or oy >= datatypes[0].shape[0]: ret.append(0)
                else: ret.append(_load(m, ox*4 + oy*datatypes[0].shape[1]*4 + j))
              data_list[i].append(ret)
          elif dtype.sz > 1:
            data_list[i] = [load(input, j) for j in range(dtype.sz)]
          else:
            data_list[i] = load(input)
        elif uop is UOps.PHI:
          for j in range(len(input[0])):
            input[0][j] = input[1][j]
          data_list[i] = input[0]
        elif uop is UOps.GEP:
          data_list[i] = input[0][arg]
        elif uop is UOps.WMMA:
          # here are the models for the WMMA instruction on the different hardware
          def wmma_helper(WARP_THREADS, K, NUM_A, NUM_B, NUM_C, a_elem, b_elem, c_map):
            assert len(input[0]) == NUM_A, f"A must have {NUM_A} elements per thread"
            assert len(input[1]) == NUM_B, f"B must have {NUM_B} elements per thread"
            assert len(input[2]) == NUM_C, f"C must have {NUM_C} elements per thread"
            assert len(flatten(input[0])) == NUM_A * warp_size, f"WMMA must have {NUM_A * warp_size} total elements for A in WMMA"
            assert len(flatten(input[1])) == NUM_B * warp_size, f"WMMA must have {NUM_B * warp_size} total elements for B in WMMA"
            assert len(flatten(input[2])) == NUM_C * warp_size, f"WMMA must have {NUM_C * warp_size} total elements for C in WMMA"
            assert warp_size > 0 and warp_size % WARP_THREADS == 0, f"must have multiples of {WARP_THREADS} warp threads"
            out = [input[2][elem_idx][:] for elem_idx in range(NUM_C)]
            for goff in range(0, warp_size, WARP_THREADS):
              for lane_id in range(WARP_THREADS):
                for elem_idx in range(NUM_C): # calculate new muls and add to acc
                  (c_i, c_j) = c_map(lane_id, elem_idx)
                  out[elem_idx][goff+lane_id] += sum(a_elem(input[0], _k, c_j, goff) * b_elem(input[1], c_i, _k, goff) for _k in range(K))
            return out

          if arg.startswith('__metal_wmma'):
            def a_b_elem(x, i, j, goff): # A (2 elements on 32 threads): row major
              return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            def c_map(lane, elem): # (i, j), C, D (2 elements on 32 threads): row major same as A/B
              return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            data_list[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif arg == '__builtin_amdgcn_wmma_f32_16x16x16_f16_w32' or arg == '__hip_wmma_f16_f16':
            def a_elem(x, i, j, goff): # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
              assert x[i][goff+j] == x[i][goff+j+16], "warp elements not duplicated properly across lanes"
              return x[i][goff+j]
            def b_elem(x, i, j, goff): # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
              return a_elem(x, j, i, goff)
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            data_list[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          else:
            raise Exception(f"unimplemented tensor core {arg}")
        elif uop is UOps.ALU:
          assert all_same([len(x) for x in input]), f"{[len(x) for x in input]} doesn't match on {arg}"
          assert all_same([dtype] + datatypes) or arg in {BinaryOps.CMPEQ, BinaryOps.CMPLT, TernaryOps.WHERE}, f"dtype mismatch on {arg}"
          data_list[i] = [exec_alu(arg, dtype, p) for p in zip(*input)]
        assert i in data_list, (uop, dtype, data_indices, arg)
        i += 1
    return time.perf_counter() - start_time

class PythonCompiler(Compiler):
  linearizer_opts = LinearizerOptions("METAL", has_tensor_cores=True) if getenv("EMULATE_METAL") else \
    (LinearizerOptions("HIP", has_tensor_cores=True) if getenv("EMULATE_HIP") else LinearizerOptions())
  def render(self, name:str, uops:List[UOp]) -> str:
    lops = [(u.uop, u.dtype, [uops.index(v) for v in u.vin], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator):
  def _alloc(self, size): return memoryview(bytearray(size))
  def copyin(self, dest, src:memoryview): dest[:] = src
  def copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(), PythonCompiler(), PythonProgram)