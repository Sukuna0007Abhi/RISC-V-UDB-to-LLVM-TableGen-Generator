#!/usr/bin/env python3
"""
RISC-V UDB to LLVM TableGen Generator
Automated Compiler Backend Generator
"""

import yaml
import argparse
import os
import sys
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class InstructionFormat(Enum):
    R_TYPE = "R"
    I_TYPE = "I" 
    S_TYPE = "S"
    B_TYPE = "B"
    U_TYPE = "U"
    J_TYPE = "J"

@dataclass
class RISCVInstruction:
    name: str
    opcode: str
    format: InstructionFormat
    fields: Dict[str, Any]
    encoding: str
    description: str
    extensions: List[str] = field(default_factory=list)
    operands: List[Dict] = field(default_factory=list)

@dataclass 
class RegisterClass:
    name: str
    registers: List[str]
    reg_type: str
    alignment: int = 4
    size: int = 32

class UDBParser:
    def __init__(self):
        self.instructions: List[RISCVInstruction] = []
        self.register_classes: Dict[str, RegisterClass] = {}
        
    def parse_udb_file(self, filepath: str) -> bool:
        try:
            with open(filepath, 'r') as file:
                udb_data = yaml.safe_load(file)
                
            if 'instructions' in udb_data:
                for instr_name, instr_data in udb_data['instructions'].items():
                    instruction = self._parse_instruction(instr_name, instr_data)
                    if instruction:
                        self.instructions.append(instruction)
                        
            if 'registers' in udb_data:
                self._parse_register_classes(udb_data['registers'])
                
            return True
            
        except Exception as e:
            print(f"Error parsing UDB file {filepath}: {e}")
            return False
    
    def _parse_instruction(self, name: str, data: Dict) -> Optional[RISCVInstruction]:
        try:
            fmt_str = data.get('format', 'R')
            try:
                instr_format = InstructionFormat(fmt_str)
            except ValueError:
                instr_format = InstructionFormat.R_TYPE
            
            instruction = RISCVInstruction(
                name=name.upper(),
                opcode=data.get('opcode', '0000000'),
                format=instr_format,
                fields=data.get('fields', {}),
                encoding=data.get('encoding', ''),
                description=data.get('description', ''),
                extensions=data.get('extensions', []),
                operands=data.get('operands', [])
            )
            
            return instruction
            
        except Exception as e:
            print(f"Error parsing instruction {name}: {e}")
            return None
    
    def _parse_register_classes(self, reg_data: Dict):
        self.register_classes['GPR'] = RegisterClass(
            name='GPR',
            registers=[f'X{i}' for i in range(32)],
            reg_type='i32',
            size=32
        )
        
        if 'f' in reg_data or 'float' in reg_data:
            self.register_classes['FPR32'] = RegisterClass(
                name='FPR32',
                registers=[f'F{i}' for i in range(32)],
                reg_type='f32',
                size=32
            )

class TableGenGenerator:
    def __init__(self, parser: UDBParser):
        self.parser = parser
        self.output_dir = "generated_tablegen"
        
    def generate_all(self, output_dir: str = None):
        if output_dir:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        start_time = time.time()
        print("Generating LLVM TableGen files...")
        
        self._generate_instructions_td()
        self._generate_registers_td()
        self._generate_patterns_td()
        self._generate_scheduling_td()
        self._generate_main_td()
        
        elapsed = time.time() - start_time
        print(f"Generated files in {self.output_dir}/")
        print(f"Performance: {len(self.parser.instructions)} instructions in {elapsed:.2f}s")
        print(f"Speed: {len(self.parser.instructions)/elapsed:.0f} instructions/second")
        
    def _generate_instructions_td(self):
        filepath = os.path.join(self.output_dir, "RISCVInstructions.td")
        
        with open(filepath, 'w') as f:
            f.write(self._get_file_header("Instruction Definitions"))
            f.write(self._generate_format_classes())
            f.write("\n// Instruction Definitions\n\n")
            
            for instr in self.parser.instructions:
                f.write(self._generate_instruction_def(instr))
                f.write("\n")
                
    def _generate_registers_td(self):
        filepath = os.path.join(self.output_dir, "RISCVRegisters.td") 
        
        with open(filepath, 'w') as f:
            f.write(self._get_file_header("Register Definitions"))
            
            for reg_class in self.parser.register_classes.values():
                f.write(self._generate_register_class(reg_class))
                f.write("\n")
                
    def _generate_patterns_td(self):
        filepath = os.path.join(self.output_dir, "RISCVPatterns.td")
        
        with open(filepath, 'w') as f:
            f.write(self._get_file_header("Instruction Selection Patterns"))
            
            f.write("// Arithmetic Patterns\n\n")
            f.write(self._generate_arithmetic_patterns())
            
            f.write("\n// Memory Patterns\n\n") 
            f.write(self._generate_memory_patterns())
            
    def _generate_scheduling_td(self):
        filepath = os.path.join(self.output_dir, "RISCVSchedule.td")
        
        with open(filepath, 'w') as f:
            f.write(self._get_file_header("Scheduling Model"))
            f.write(self._generate_scheduling_model())
            
    def _generate_main_td(self):
        filepath = os.path.join(self.output_dir, "RISCVTarget.td")
        
        with open(filepath, 'w') as f:
            f.write(self._get_file_header("Main Target Definition"))
            f.write('''
include "llvm/Target/Target.td"

include "RISCVRegisters.td"
include "RISCVInstructions.td" 
include "RISCVPatterns.td"
include "RISCVSchedule.td"

def RISCV : Target {
  let InstructionSet = RISCVInstrInfo;
  let AssemblyWriters = [RISCVAsmWriter];
  let AssemblyParsers = [RISCVAsmParser];
}
''')
    
    def _get_file_header(self, description: str) -> str:
        return f'''//===-- RISC-V {description} --------*- tablegen -*-===//
//
// Automatically generated from RISC-V UDB specifications
// Generated by: UDB to LLVM TableGen Generator
//
//===----------------------------------------------------------------------===//

'''
    
    def _generate_format_classes(self) -> str:
        return '''// Instruction Format Classes

class RISCVInstFormat<bits<3> val> {
  bits<3> Value = val;
}

def R_FORMAT : RISCVInstFormat<0>;
def I_FORMAT : RISCVInstFormat<1>;  
def S_FORMAT : RISCVInstFormat<2>;
def B_FORMAT : RISCVInstFormat<3>;
def U_FORMAT : RISCVInstFormat<4>;
def J_FORMAT : RISCVInstFormat<5>;

class RISCVInst<bits<7> opcode, RISCVInstFormat format, dag outs, dag ins,
               string asmstr, list<dag> pattern = []>
    : Instruction {
  field bits<32> Inst;
  let Namespace = "RISCV";
  
  bits<7> Opcode = opcode;
  RISCVInstFormat Format = format;
  
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  
  let Inst{6-0} = Opcode;
}

'''
    
    def _generate_instruction_def(self, instr: RISCVInstruction) -> str:
        operands = self._get_operands_for_format(instr.format)
        opcode_bits = self._parse_opcode_bits(instr.opcode)
        
        return f'''def {instr.name} : RISCVInst<{opcode_bits}, {instr.format.value}_FORMAT,
                   {operands['outs']}, {operands['ins']},
                   "{instr.name.lower()} {operands['asm']}", []> {{
  let hasSideEffects = 0;
  let mayLoad = {str(self._is_load_instr(instr)).lower()};
  let mayStore = {str(self._is_store_instr(instr)).lower()};
}}'''
    
    def _get_operands_for_format(self, format: InstructionFormat) -> Dict[str, str]:
        formats = {
            InstructionFormat.R_TYPE: {
                'outs': '(outs GPR:$rd)',
                'ins': '(ins GPR:$rs1, GPR:$rs2)', 
                'asm': '$rd, $rs1, $rs2'
            },
            InstructionFormat.I_TYPE: {
                'outs': '(outs GPR:$rd)',
                'ins': '(ins GPR:$rs1, simm12:$imm12)',
                'asm': '$rd, $rs1, $imm12'  
            },
            InstructionFormat.S_TYPE: {
                'outs': '(outs)',
                'ins': '(ins GPR:$rs2, GPR:$rs1, simm12:$imm12)',
                'asm': '$rs2, ${imm12}(${rs1})'
            },
            InstructionFormat.B_TYPE: {
                'outs': '(outs)', 
                'ins': '(ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12)',
                'asm': '$rs1, $rs2, $imm12'
            },
            InstructionFormat.U_TYPE: {
                'outs': '(outs GPR:$rd)',
                'ins': '(ins uimm20:$imm20)', 
                'asm': '$rd, $imm20'
            },
            InstructionFormat.J_TYPE: {
                'outs': '(outs GPR:$rd)',
                'ins': '(ins simm21_lsb0:$imm20)',
                'asm': '$rd, $imm20'
            }
        }
        return formats.get(format, formats[InstructionFormat.R_TYPE])
    
    def _parse_opcode_bits(self, opcode: str) -> str:
        if opcode.startswith('0b'):
            return f"0b{opcode[2:].zfill(7)}"
        elif opcode.startswith('0x'):
            val = int(opcode, 16)
            return f"0b{format(val, '07b')}"
        else:
            return "0b0110011"
            
    def _is_load_instr(self, instr: RISCVInstruction) -> bool:
        return 'load' in instr.name.lower() or instr.name.lower().startswith('l')
        
    def _is_store_instr(self, instr: RISCVInstruction) -> bool: 
        return 'store' in instr.name.lower() or instr.name.lower().startswith('s')
    
    def _generate_register_class(self, reg_class: RegisterClass) -> str:
        registers = ', '.join(reg_class.registers)
        
        return f'''let Namespace = "RISCV" in {{
  foreach i = 0-31 in {{
    def {reg_class.name}#i : Register<"{reg_class.name.lower()}"#i>, DwarfRegNum<[i]>;
  }}
}}

def {reg_class.name} : RegisterClass<"RISCV", [{reg_class.reg_type}], {reg_class.size}, (add
  {registers}
)>;
'''
    
    def _generate_arithmetic_patterns(self) -> str:
        return '''def : Pat<(add GPR:$rs1, GPR:$rs2), (ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(sub GPR:$rs1, GPR:$rs2), (SUB GPR:$rs1, GPR:$rs2)>;
def : Pat<(and GPR:$rs1, GPR:$rs2), (AND GPR:$rs1, GPR:$rs2)>;
def : Pat<(or GPR:$rs1, GPR:$rs2), (OR GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor GPR:$rs1, GPR:$rs2), (XOR GPR:$rs1, GPR:$rs2)>;

def : Pat<(addi GPR:$rs1, simm12:$imm), (ADDI GPR:$rs1, simm12:$imm)>;
def : Pat<(andi GPR:$rs1, simm12:$imm), (ANDI GPR:$rs1, simm12:$imm)>;
def : Pat<(ori GPR:$rs1, simm12:$imm), (ORI GPR:$rs1, simm12:$imm)>;
'''
    
    def _generate_memory_patterns(self) -> str:
        return '''def : Pat<(i32 (load (add GPR:$rs1, simm12:$imm))), (LW GPR:$rs1, simm12:$imm)>;
def : Pat<(i16 (load (add GPR:$rs1, simm12:$imm))), (LH GPR:$rs1, simm12:$imm)>;
def : Pat<(i8 (load (add GPR:$rs1, simm12:$imm))), (LB GPR:$rs1, simm12:$imm)>;

def : Pat<(store GPR:$rs2, (add GPR:$rs1, simm12:$imm)), 
          (SW GPR:$rs2, GPR:$rs1, simm12:$imm)>;
def : Pat<(truncstorei16 GPR:$rs2, (add GPR:$rs1, simm12:$imm)),
          (SH GPR:$rs2, GPR:$rs1, simm12:$imm)>;  
def : Pat<(truncstorei8 GPR:$rs2, (add GPR:$rs1, simm12:$imm)),
          (SB GPR:$rs2, GPR:$rs1, simm12:$imm)>;
'''
    
    def _generate_scheduling_model(self) -> str:
        return '''// Scheduling Model

def RISCVModel : SchedMachineModel {
  let IssueWidth = 1;
  let MicroOpBufferSize = 0;
  let LoopMicroOpBufferSize = 0;
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let CompleteModel = 0;
}

def ALU : ProcResource<1>;
def LSU : ProcResource<1>;
def MUL : ProcResource<1>;
def DIV : ProcResource<1>;

def : WriteRes<WriteIALU, [ALU]> { let Latency = 1; }
def : WriteRes<WriteIMul, [MUL]> { let Latency = 3; }  
def : WriteRes<WriteIDiv, [DIV]> { let Latency = 20; }
def : WriteRes<WriteLd, [LSU]> { let Latency = 3; }
def : WriteRes<WriteSt, [LSU]> { let Latency = 1; }
'''

def create_sample_udb():
    sample_udb = {
        'instructions': {
            'add': {
                'format': 'R',
                'opcode': '0110011',
                'encoding': 'funct7[31:25] rs2[24:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]',
                'description': 'Add registers',
                'extensions': ['I'],
                'operands': [
                    {'name': 'rd', 'type': 'register'},
                    {'name': 'rs1', 'type': 'register'},
                    {'name': 'rs2', 'type': 'register'}
                ]
            },
            'addi': {
                'format': 'I', 
                'opcode': '0010011',
                'encoding': 'imm[31:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]',
                'description': 'Add immediate',
                'extensions': ['I'],
                'operands': [
                    {'name': 'rd', 'type': 'register'},
                    {'name': 'rs1', 'type': 'register'}, 
                    {'name': 'imm', 'type': 'immediate'}
                ]
            },
            'sub': {
                'format': 'R',
                'opcode': '0110011',
                'description': 'Subtract registers',
                'extensions': ['I']
            },
            'lw': {
                'format': 'I',
                'opcode': '0000011', 
                'encoding': 'imm[31:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]',
                'description': 'Load word',
                'extensions': ['I'],
                'operands': [
                    {'name': 'rd', 'type': 'register'},
                    {'name': 'rs1', 'type': 'register'},
                    {'name': 'imm', 'type': 'immediate'}
                ]
            },
            'sw': {
                'format': 'S',
                'opcode': '0100011',
                'encoding': 'imm[31:25] rs2[24:20] rs1[19:15] funct3[14:12] imm[11:7] opcode[6:0]', 
                'description': 'Store word',
                'extensions': ['I'],
                'operands': [
                    {'name': 'rs2', 'type': 'register'},
                    {'name': 'rs1', 'type': 'register'},
                    {'name': 'imm', 'type': 'immediate'}
                ]
            },
            'fadd_s': {
                'format': 'R',
                'opcode': '1010011',
                'description': 'Floating-point add single',
                'extensions': ['F']
            },
            'fmul_s': {
                'format': 'R', 
                'opcode': '1010011',
                'description': 'Floating-point multiply single',
                'extensions': ['F']
            }
        },
        'registers': {
            'gpr': {
                'count': 32,
                'width': 32,
                'names': [f'x{i}' for i in range(32)]
            },
            'float': {
                'count': 32,
                'width': 32,
                'names': [f'f{i}' for i in range(32)]
            }
        }
    }
    
    with open('sample_udb.yaml', 'w') as f:
        yaml.dump(sample_udb, f, default_flow_style=False)
    
    return 'sample_udb.yaml'

def main():
    parser = argparse.ArgumentParser(
        description='RISC-V UDB to LLVM TableGen Generator',
        epilog='Example: python3 udb_tablegen_gen.py --input sample_udb.yaml --output generated/'
    )
    parser.add_argument('--input', '-i', 
                       help='Input UDB YAML file')
    parser.add_argument('--output', '-o', default='generated_tablegen',
                       help='Output directory for generated files')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample UDB file for testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.create_sample:
        sample_file = create_sample_udb()
        print(f"Created sample UDB file: {sample_file}")
        print("Now run: python3 udb_tablegen_gen.py -i sample_udb.yaml -o generated/")
        return 0
    
    if not args.input:
        print("Error: --input required (or use --create-sample)")
        return 1
        
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        print("Use --create-sample to generate a sample UDB file")
        return 1
    
    print(f"Parsing UDB file: {args.input}")
    udb_parser = UDBParser()
    
    if not udb_parser.parse_udb_file(args.input):
        print("Failed to parse UDB file")
        return 1
        
    if args.verbose:
        print(f"Parsed {len(udb_parser.instructions)} instructions")
        print(f"Found {len(udb_parser.register_classes)} register classes")
    
    generator = TableGenGenerator(udb_parser)
    generator.generate_all(args.output)
    
    print(f"\nSuccessfully generated LLVM TableGen files in {args.output}/")
    print("\nGenerated files:")
    for file in ['RISCVTarget.td', 'RISCVInstructions.td', 'RISCVRegisters.td', 
                 'RISCVPatterns.td', 'RISCVSchedule.td']:
        print(f"  - {file}")
    
    print(f"\nStatistics:")
    print(f"  - Instructions processed: {len(udb_parser.instructions)}")
    print(f"  - Register classes: {len(udb_parser.register_classes)}")
    print(f"  - Estimated manual effort saved: ~2-3 weeks")
    print(f"\nNext: Integrate with LLVM build system")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())