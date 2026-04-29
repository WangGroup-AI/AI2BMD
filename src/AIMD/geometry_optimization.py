import os
import shutil
import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS, BFGS, BFGSLineSearch
from ase.optimize.minimahopping import MinimaHopping
from ase.constraints import Hookean
from AIMD import arguments 
from AIMD.protein import Protein
from Calculators.device_strategy import DeviceStrategy
from Calculators.fragment import FragmentCalculator
from Calculators.visnet_calculator import ViSNetCalculator
from utils.pdb import read_protein


import numpy as np
from numpy.linalg import eigh

import numpy as np
from ase.optimize import BFGS
import time
args = arguments.init()
class AI2BMDOptimizer:
    def __init__(self, atoms, fmax=0.015, max_cycles=100, max_step_size=0.1):
        self.atoms = atoms
        self.fmax = fmax
        self.max_cycles=max_cycles
        self.max_step_size = max_step_size        
        self.traj_file = "optimization_trace.xyz"
        self.best_file = "best_structure.xyz"

    def run(self):
        all_best_f = 999.0
        for cycle in range(self.max_cycles):
            print(f"\n🚀 [Cycle {cycle+1}] 启动精细优化 (BFGS)...")
            
            curr_max_step_size = self.max_step_size
            if cycle >=0:
                opt = LBFGS(self.atoms, memory=20, alpha=70.0, maxstep=curr_max_step_size)
                #opt = BFGSLineSearch(self.atoms, alpha=70.0, maxstep=curr_max_step_size)
                #opt = BFGS(self.atoms, alpha=70.0, maxstep=self.max_step_size)
            else:
                opt = BFGSLineSearch(self.atoms, alpha=40.0, maxstep=curr_max_step_size)

            stagnant_steps = 0
            best_f = 999.0
            
            for step in opt.irun(fmax=self.fmax,steps=100):
                write(self.traj_file, self.atoms, append=True)
                curr_f = self.cal_fmax()
                
                if curr_f < best_f:
                    best_f = curr_f
                    curr_max_step_size = curr_max_step_size/2
                    print(f"✨更新了局部极小值(fmax={best_f:.4f})")
                    if best_f < all_best_f:
                        all_best_f = best_f
                        #curr_max_step_size = curr_max_step_size/10
                        write(self.best_file, self.atoms)
                        print(f"✨更新了整体极小值(fmax={all_best_f:.4f})")

                # 检查是否进入平台期
                if (curr_f-best_f) > 1e-6:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0
                
                """
                # 如果停滞超过 15 步，强制跳出
                if stagnant_steps >= 50: #default = 100
                    print(f"⚠️  检测到 BFGS 陷入局部极小值 (fmax={best_f:.4f})，触发 SGD 破局逻辑...")
                    break
                """
            else:
                # 如果 irun 正常结束（即达到 fmax），整个优化完成
                print("🎉 结构已成功收敛！")
                return True
            
            print(f"✨当前整体极小值(fmax={all_best_f:.4f})")
            # --- 进入 SGD 破局阶段 ---
            print(f"🛠️  [Breakout] 正在进行强制坐标重构...")
            self.shake_and_push(steps=20, lr=0.01) #default steps=20 lr = 0.01

        print("❌ 达到最大循环次数，仍未完全收敛，建议检查 ML 势函数在该构型下的受力合理性。")
        return False
    
    def cal_fmax(self):
        forces = self.atoms.get_forces()
        fmax = np.sqrt((forces**2).sum(axis=1).max())
        return fmax
    
    def shake_and_push(self, steps=20, lr=0.01): # 调低初始 lr
        for i in range(steps):
            forces = self.atoms.get_forces()
            fmax = self.cal_fmax()
            actual_lr = lr
            if fmax > 100.0:
                actual_lr = lr / 100.0
            elif fmax > 10.0:
                actual_lr = lr / 10.0
                
            pos = self.atoms.get_positions()
            move = actual_lr * forces
            
            max_allowed = 0.05 
            lengths = np.linalg.norm(move, axis=1)
            mask = lengths > max_allowed
            if mask.any():
                move[mask] *= (max_allowed / lengths[mask])[:, np.newaxis]
            
            noise = np.random.normal(0, 0.005, pos.shape) #default=0.001
            
            self.atoms.set_positions(pos + move + noise)
            
            if i % 10 == 0:
                print(f"   - SGD 推送中... Step {i}, 当前 fmax: {fmax:.4f}")




class ProteinOptimizer:
    def __init__(
        self, prot: Protein, log_path: str, fmax: float = 0.015
    ) -> None:
        self.prot = prot
        self.log_path = log_path
        self.optimization_save_path = os.path.join(log_path, "OptimizationResults")
        os.makedirs(self.optimization_save_path, exist_ok=True)
        
        self.nowat_pdb = self.prot.nowater_PDB_path
        self.fmax = fmax
        self.prot.set_pbc(True)
        self.qmcalc = None

    def get_qm_idx(self):
        """获取所有原子的索引用于限制或分析"""
        return list(range(len(read_protein(self.nowat_pdb))))

    def need_fragmentation(self):
        """判断当前计算器是否需要进行片段划分"""
        return isinstance(self.qmcalc, FragmentCalculator)

    def initialize_fragcalc(self):
        """初始化片段计算器并分配设备计算资源"""
        if self.need_fragmentation():
            # 执行片段化逻辑
            self.qmcalc.bonded_calculator.fragment_method.fragment(self.prot)
            self.qmcalc.nonbonded_calculator.set_parameters(self.prot)

            # 获取片段的起始和结束索引
            start, end = self.prot.fragments_start, self.prot.fragments_end
        else:
            # 非片段化模式（如全局 ViSNet 或单体计算）
            start, end = [0], [len(self.prot)]

        # 根据片段划分设置 GPU/CPU 工作负载分配
        DeviceStrategy.set_work_partitions(start, end)

    def make_calculator(self, is_root_calc: bool, **kwargs):
        """
        工厂方法：根据配置参数决定使用哪种 AI 力场或片段计算器
        """
        mode = arguments.get().mode
        if mode == "fragment":
            return FragmentCalculator(is_root_calc=is_root_calc, **kwargs)
        elif mode == "visnet":
            # 假设 ViSNetCalculator 继承自或兼容 FragmentCalculator 逻辑
            return ViSNetCalculator(is_root_calc=is_root_calc, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Please check your arguments.")

    def set_calculator(self, **kwargs) -> None:
        """设置计算器并执行初始化"""
        # 切换到工作目录，防止临时文件污染根目录
        os.chdir(self.optimization_save_path)
        
        self.qmcalc = self.make_calculator(is_root_calc=True, **kwargs)
        self.prot.set_calculator(self.qmcalc)
        
        # 执行片段初始化和 DeviceStrategy 分配
        self.initialize_fragcalc()

    def optimize(self, prot_name: str, max_cycles: int = 1000, max_step_size: float = 0.1):
        print(f"--- Starting Structure Optimization for {prot_name} ---")
        print(f"Mode: {arguments.get().mode} | Target fmax: {self.fmax} eV/A")
        
        opt = AI2BMDOptimizer(atoms=self.prot, fmax=self.fmax, max_cycles=max_cycles, max_step_size=max_step_size)

        try:
            opt.run()
            print("Optimization successfully converged!")

            if args.vib:
                from ase.vibrations import Vibrations
                vib_name = 'vib_best_structure'
                vib = Vibrations(self.prot, name=vib_name)
                
                print("\n--- 开始计算 Hessian 矩阵 (有限位移法) ---")
                vib.run()

                #输出频率汇总
                print("\n--- 振动频率结果 ---")
                vib.summary()

        except Exception as e:
            print(f"Optimization encountered an error: {e}")
