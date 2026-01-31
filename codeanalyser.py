"""
LLM-Ready Analyzer - Streamlined output for AI test generation
Combines enhanced_analyzer.py + advanced_cfg_engine.py into one optimized format
"""
from __future__ import annotations
import ast
import json
import symtable
import logging
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for code analyzer behavior"""
    max_loop_iterations: int = 2
    max_path_depth: int = 50
    extract_string_literals: bool = True
    enable_short_circuit_detection: bool = True
    verbose_logging: bool = False


@dataclass
class LLMTestGenerationPayload:
    """The exact format LLMs need to generate test cases"""
    function_name: str
    signature: Dict[str, Any]
    execution_paths: List[Dict[str, Any]]
    test_guidance: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps({
            "function": self.function_name,
            "signature": self.signature,
            "paths": self.execution_paths,
            "guidance": self.test_guidance
        }, indent=2)


class LLMReadyAnalyzer:
    """
    Unified analyzer that outputs LLM-optimized JSON for test generation
    """
    
    def __init__(self, source_code: str, function_name: str, config: Optional[AnalyzerConfig] = None):
        # Validate inputs
        if not source_code or not isinstance(source_code, str):
            raise ValueError("source_code must be a non-empty string")
        if not function_name or not isinstance(function_name, str):
            raise ValueError("function_name must be a non-empty string")
        
        self.config = config or AnalyzerConfig()
        self.source = source_code
        self.function_name = function_name
        
        try:
            self.tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {str(e)}")
        
        self.func_node = None
        self.entry_node_id = None  # FIX BUG 1: Store entry ID
        self.if_condition_stack = []  # FIX BUG 4: Track conditions for exceptions
        self.path_depth_counter = 0
        
        # Find target function
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                self.func_node = node
                break
        
        if not self.func_node:
            raise ValueError(f"Function '{function_name}' not found")
        
        # Initialize symbol table
        try:
            self.symtable = symtable.symtable(source_code, "<string>", "exec")
        except (SyntaxError, Exception) as e:
            if self.config.verbose_logging:
                logger.warning(f"Could not build symbol table for {function_name}: {str(e)}")
            self.symtable = None
    
    def analyze_for_llm(self) -> LLMTestGenerationPayload:
        """Generate LLM-ready output for test generation"""
        
        # 1. Extract signature
        signature = self._extract_signature()
        
        # 2. Build CFG and enumerate paths
        cfg_nodes = self._build_cfg()
        paths = self._enumerate_paths(cfg_nodes)
        
        # 3. Generate test guidance
        guidance = self._generate_test_guidance(cfg_nodes, paths)
        
        return LLMTestGenerationPayload(
            function_name=self.function_name,
            signature=signature,
            execution_paths=paths,
            test_guidance=guidance
        )
    
    def _extract_signature(self) -> Dict[str, Any]:
        """Extract function signature for LLM"""
        params = []
        for arg in self.func_node.args.args:
            param_info = {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else "Any",
                "required": True
            }
            params.append(param_info)
        
        # Handle defaults
        defaults = self.func_node.args.defaults
        if defaults:
            for i, default in enumerate(defaults):
                param_idx = len(params) - len(defaults) + i
                params[param_idx]["required"] = False
                params[param_idx]["default"] = ast.unparse(default)
        
        return {
            "name": self.function_name,
            "parameters": params,
            "return_type": ast.unparse(self.func_node.returns) if self.func_node.returns else "Any",
            "docstring": ast.get_docstring(self.func_node)
        }
    
    def _build_cfg(self) -> List[Dict]:
        """Build REAL CFG with proper edges and control flow"""
        self.cfg_nodes = {}
        self.node_counter = 0
        
        # Entry node
        entry = self._create_node("entry", f"def {self.function_name}(...)")
        self.entry_node_id = entry["id"]  # FIX BUG 1: Store entry ID explicitly
        
        # Build CFG by processing statements
        last_node = entry
        for stmt in self.func_node.body:
            last_node = self._process_statement(stmt, last_node)
        
        # If no explicit return, add implicit return None
        if last_node and last_node["type"] not in ["return", "raise"]:
            implicit_return = self._create_node("return", "return None", value="None")
            self._connect(last_node, implicit_return)
        
        # Convert to list format
        return list(self.cfg_nodes.values())
    
    def _create_node(self, node_type: str, code: str, **kwargs) -> Dict:
        """Create a CFG node with unique ID"""
        self.node_counter += 1
        node_id = f"n{self.node_counter}"
        
        node = {
            "id": node_id,
            "type": node_type,
            "code": code,
            "next": [],
            **kwargs
        }
        self.cfg_nodes[node_id] = node
        return node
    
    def _connect(self, from_node: Dict, to_node: Dict):
        """Create CFG edge"""
        if to_node["id"] not in from_node["next"]:
            from_node["next"].append(to_node["id"])
    
    def _process_statement(self, stmt, predecessor: Dict) -> Dict:
        """Process a statement and return the last node"""
        
        if isinstance(stmt, ast.Return):
            node = self._create_node(
                "return",
                ast.unparse(stmt),
                value=ast.unparse(stmt.value) if stmt.value else "None"
            )
            self._connect(predecessor, node)
            return node  # Terminal
        
        elif isinstance(stmt, ast.Raise):
            node = self._create_node(
                "raise",
                ast.unparse(stmt),
                exception=ast.unparse(stmt.exc),
                constraint=self._extract_exception_constraint(stmt)
            )
            self._connect(predecessor, node)
            return node  # Terminal
        
        elif isinstance(stmt, ast.Assign):
            targets = [ast.unparse(t) for t in stmt.targets]
            node = self._create_node(
                "assign",
                ast.unparse(stmt),
                variables=targets,
                value=ast.unparse(stmt.value)
            )
            self._connect(predecessor, node)
            return node
        
        elif isinstance(stmt, ast.AugAssign):
            # Handle += -= *= /= etc.
            target = ast.unparse(stmt.target)
            node = self._create_node(
                "assign",
                ast.unparse(stmt),
                variables=[target],
                value=ast.unparse(stmt.value)
            )
            self._connect(predecessor, node)
            return node
        
        elif isinstance(stmt, ast.If):
            # Decision node
            condition = ast.unparse(stmt.test)
            if_node = self._create_node(
                "decision",
                f"if {condition}",
                condition=condition,
                condition_ast=stmt.test
            )
            self._connect(predecessor, if_node)
            
            # Push condition onto stack for exception tracking
            self.if_condition_stack.append((condition, stmt.test))
            
            # True branch
            true_last = if_node
            true_first = None
            for s in stmt.body:
                nodes_before = len(self.cfg_nodes)
                true_last = self._process_statement(s, true_last)
                # Find first node created in this branch
                if true_first is None and len(self.cfg_nodes) > nodes_before:
                    new_node_ids = list(self.cfg_nodes.keys())[nodes_before:]
                    if new_node_ids:
                        true_first = self.cfg_nodes[new_node_ids[0]]
            
            # Handle empty body case
            if true_first is None:
                true_first = self._create_node("nop", "pass")
                self._connect(if_node, true_first)
                true_last = true_first
            
            # False branch (else)
            false_last = if_node
            false_first = None
            if stmt.orelse:
                for s in stmt.orelse:
                    nodes_before = len(self.cfg_nodes)
                    false_last = self._process_statement(s, false_last)
                    # Find first node created in else branch
                    if false_first is None and len(self.cfg_nodes) > nodes_before:
                        new_node_ids = list(self.cfg_nodes.keys())[nodes_before:]
                        if new_node_ids:
                            false_first = self.cfg_nodes[new_node_ids[0]]
            
            # Pop condition stack
            self.if_condition_stack.pop()
            
            # FIX BUG 2: Explicitly label true/false edges
            if true_first:
                if_node["true_next"] = true_first["id"]
            if false_first:
                if_node["false_next"] = false_first["id"]
            elif not stmt.orelse:
                # No else: false branch falls through
                if_node["false_next"] = "fall_through"
            
            # Merge node - only if at least one branch is non-terminal
            if true_last["type"] not in ["return", "raise"] or false_last["type"] not in ["return", "raise"]:
                merge = self._create_node("merge", "merge point")
                
                # Connect non-terminal branches to merge
                if true_last["type"] not in ["return", "raise"]:
                    self._connect(true_last, merge)
                if false_last["type"] not in ["return", "raise"]:
                    self._connect(false_last, merge)
                
                return merge
            else:
                # Both branches are terminal - no merge needed
                return if_node
        
        elif isinstance(stmt, ast.For):
            loop_var = ast.unparse(stmt.target)
            iterable = ast.unparse(stmt.iter)
            
            # Loop header
            loop_header = self._create_node(
                "loop",
                f"for {loop_var} in {iterable}",
                loop_variable=loop_var,
                iterable=iterable
            )
            self._connect(predecessor, loop_header)
            
            # Loop body - process statements
            body_first = None
            body_last = loop_header
            
            if stmt.body:
                # Record node count before processing
                nodes_before = len(self.cfg_nodes)
                
                # Process first statement
                body_last = self._process_statement(stmt.body[0], body_last)
                
                # Find the first node created (should be right after loop_header)
                for node_id in list(self.cfg_nodes.keys())[nodes_before:]:
                    node = self.cfg_nodes[node_id]
                    if node != loop_header:
                        body_first = node
                        break
                
                # Process remaining statements
                for s in stmt.body[1:]:
                    body_last = self._process_statement(s, body_last)
            
            # Loop exit
            loop_exit = self._create_node("loop_exit", "exit loop")
            
            # FIX BUG 3: Explicitly label loop edges
            if body_first:
                loop_header["body_next"] = body_first["id"]
                # Back edge to loop header
                if body_last["type"] not in ["return", "raise"]:
                    self._connect(body_last, loop_header)
            
            loop_header["exit_next"] = loop_exit["id"]
            self._connect(loop_header, loop_exit)
            
            return loop_exit
        
        elif isinstance(stmt, ast.While):
            # While loop - similar to for loop
            condition = ast.unparse(stmt.test)
            while_node = self._create_node(
                "loop",
                f"while {condition}",
                loop_variable="(while condition)",
                iterable=condition
            )
            self._connect(predecessor, while_node)
            
            # Process body
            body_first = None
            body_last = while_node
            for s in stmt.body:
                nodes_before = len(self.cfg_nodes)
                body_last = self._process_statement(s, body_last)
                if body_first is None and len(self.cfg_nodes) > nodes_before:
                    new_node_ids = list(self.cfg_nodes.keys())[nodes_before:]
                    if new_node_ids:
                        body_first = self.cfg_nodes[new_node_ids[0]]
            
            if body_first:
                while_node["body_next"] = body_first["id"]
                if body_last["type"] not in ["return", "raise"]:
                    self._connect(body_last, while_node)  # Back edge
            
            loop_exit = self._create_node("loop_exit", "exit loop")
            while_node["exit_next"] = loop_exit["id"]
            self._connect(while_node, loop_exit)
            return loop_exit
        
        elif isinstance(stmt, ast.Try):
            # Try/Except blocks
            body_first = None
            body_last = predecessor
            for s in stmt.body:
                nodes_before = len(self.cfg_nodes)
                body_last = self._process_statement(s, body_last)
                if body_first is None and len(self.cfg_nodes) > nodes_before:
                    new_node_ids = list(self.cfg_nodes.keys())[nodes_before:]
                    if new_node_ids:
                        body_first = self.cfg_nodes[new_node_ids[0]]
            
            # Process exception handlers
            for handler in stmt.handlers:
                handler_first = None
                handler_last = body_last
                for s in handler.body:
                    nodes_before = len(self.cfg_nodes)
                    handler_last = self._process_statement(s, handler_last)
                    if handler_first is None and len(self.cfg_nodes) > nodes_before:
                        new_node_ids = list(self.cfg_nodes.keys())[nodes_before:]
                        if new_node_ids:
                            handler_first = self.cfg_nodes[new_node_ids[0]]
            
            return body_last
        
        elif isinstance(stmt, ast.With):
            # With statement - treat as single block
            context_expr = ", ".join(ast.unparse(item.context_expr) for item in stmt.items)
            with_node = self._create_node(
                "expr",
                f"with {context_expr}:"
            )
            self._connect(predecessor, with_node)
            
            last = with_node
            for s in stmt.body:
                last = self._process_statement(s, last)
            
            return last
        
        elif isinstance(stmt, ast.Expr):
            # Expression statement (e.g., function call)
            node = self._create_node(
                "expr",
                ast.unparse(stmt.value)
            )
            self._connect(predecessor, node)
            return node
        
        # Default: pass through
        return predecessor
    
    def _extract_exception_constraint(self, raise_stmt: ast.Raise) -> str:
        """Extract the REAL condition that triggers this exception - FIX BUG 4"""
        if not self.if_condition_stack:
            return "unconditional"  # Raised without any if guard
        
        # The exception is guarded by the current if conditions
        # Build the constraint from the stack
        conditions = []
        for cond_str, cond_ast in self.if_condition_stack:
            conditions.append(cond_str)
        
        # Join with AND (all conditions must be true to reach this raise)
        if conditions:
            return " and ".join(f"({c})" for c in conditions)
        
        return "unconditional"
    
    def _enumerate_paths(self, cfg_nodes: List[Dict]) -> List[Dict[str, Any]]:
        """Enumerate REAL executable paths with proper constraints"""
        paths = []
        node_map = {n["id"]: n for n in cfg_nodes}
        
        def dfs_paths(node_id: str, current_path: List[str], constraints: List[Dict], 
                     visited_on_path: Set[str], loop_iterations: Dict[str, int]):
            
            # Path depth check
            self.path_depth_counter += 1
            if self.path_depth_counter > self.config.max_path_depth:
                if self.config.verbose_logging:
                    logger.warning(f"Max path depth {self.config.max_path_depth} reached")
                self.path_depth_counter -= 1
                return
            
            # Loop detection with bound
            if node_id in visited_on_path:
                node = node_map.get(node_id)
                if node and node["type"] == "loop":
                    # Limit loop iterations
                    loop_iter = loop_iterations.get(node_id, 0)
                    if loop_iter >= self.config.max_loop_iterations:
                        self.path_depth_counter -= 1
                        return
                    loop_iterations = {**loop_iterations, node_id: loop_iter + 1}
                else:
                    self.path_depth_counter -= 1
                    return  # Cycle detected, stop
            
            self.path_depth_counter -= 1
            
            node = node_map.get(node_id)
            if not node:
                return
            
            current_path = current_path + [node_id]
            visited_on_path = visited_on_path | {node_id}
            
            # Terminal nodes
            if node["type"] in ["return", "raise"]:
                path_constraints = self._simplify_constraints(constraints)
                outcome_value = node.get("value") or node.get("exception")
                outcome = {
                    "type": node["type"],
                    "value": outcome_value
                }
                
                # GAP 2: Add return structure analysis
                if node["type"] == "return" and outcome_value:
                    outcome["structure"] = self._analyze_return_structure(outcome_value)
                
                paths.append({
                    "path_id": f"path_{len(paths) + 1}",
                    "nodes": current_path[:],
                    "constraints": [c["expr"] for c in path_constraints],
                    "constraint_details": path_constraints,
                    "outcome": outcome,
                    "state": self._compute_path_state(current_path, node_map)
                })
                return
            
            # Decision nodes - use EXPLICIT true/false edges (FIX BUG 2)
            if node["type"] == "decision":
                true_next = node.get("true_next")
                false_next = node.get("false_next")
                
                # True branch
                if true_next and true_next != "fall_through":
                    true_constraint = {
                        "expr": node["condition"],
                        "node": node_id,
                        "branch": "true",
                        "ast": node.get("condition_ast")
                    }
                    dfs_paths(true_next, current_path, 
                             constraints + [true_constraint], 
                             visited_on_path, loop_iterations)
                
                # False branch
                if false_next:
                    if false_next == "fall_through":
                        # No explicit else - continue with next nodes
                        false_constraint = {
                            "expr": f"not ({node['condition']})",
                            "node": node_id,
                            "branch": "false",
                            "ast": node.get("condition_ast")
                        }
                        # Find merge or next statement
                        for next_id in node.get("next", []):
                            if next_id != true_next:
                                dfs_paths(next_id, current_path, 
                                         constraints + [false_constraint], 
                                         visited_on_path, loop_iterations)
                    else:
                        false_constraint = {
                            "expr": f"not ({node['condition']})",
                            "node": node_id,
                            "branch": "false",
                            "ast": node.get("condition_ast")
                        }
                        dfs_paths(false_next, current_path, 
                                 constraints + [false_constraint], 
                                 visited_on_path, loop_iterations)
                return
            
            # Loop nodes - use EXPLICIT body/exit edges (FIX BUG 3)
            if node["type"] == "loop":
                body_next = node.get("body_next")
                exit_next = node.get("exit_next")
                
                # Path 1: Zero iterations (skip loop)
                if exit_next:
                    zero_iter_constraint = {
                        "expr": f"len({node['iterable']}) == 0",
                        "node": node_id,
                        "branch": "skip",
                        "loop_iterations": 0
                    }
                    dfs_paths(exit_next, current_path, 
                             constraints + [zero_iter_constraint],
                             visited_on_path, loop_iterations)
                
                # Path 2: Enter loop body (1+ iterations)
                if body_next:
                    iter_constraint = {
                        "expr": f"len({node['iterable']}) > 0",
                        "node": node_id,
                        "branch": "enter",
                        "loop_iterations": loop_iterations.get(node_id, 0) + 1
                    }
                    dfs_paths(body_next, current_path, 
                             constraints + [iter_constraint],
                             visited_on_path, loop_iterations)
                return
            
            # Regular nodes - follow edges
            for next_id in node.get("next", []):
                dfs_paths(next_id, current_path, constraints, 
                         visited_on_path, loop_iterations)
        
        # FIX BUG 1: Use stored entry node ID
        if self.entry_node_id:
            dfs_paths(self.entry_node_id, [], [], set(), {})
        
        return paths
    
    def _simplify_constraints(self, constraints: List[Dict]) -> List[Dict]:
        """Simplify and deduplicate constraints with basic normalization (GAP 1)"""
        # Remove duplicates while preserving order
        seen = set()
        simplified = []
        for c in constraints:
            expr = c["expr"]
            
            # Basic normalization: not (not X) -> X
            normalized = expr.replace("not (not ", "")
            # Remove double parens
            while "((" in normalized and "))" in normalized:
                normalized = normalized.replace("((", "(").replace("))", ")")
            
            if normalized not in seen:
                seen.add(normalized)
                c_copy = {
                    "expr": normalized,
                    "node": c.get("node"),
                    "branch": c.get("branch")
                }
                # Detect short-circuit if AST available (GAP 3 FIX)
                # Store short-circuit info but don't include AST in JSON
                if "ast" in c and c["ast"]:
                    c_copy["short_circuit"] = self._detect_short_circuit(c["ast"])
                simplified.append(c_copy)
        return simplified
    
    def _detect_short_circuit(self, condition_ast: ast.AST) -> Dict[str, Any]:
        """Detect short-circuit boolean operators using AST (GAP 3 FIX)"""
        if isinstance(condition_ast, ast.BoolOp):
            op_type = "and" if isinstance(condition_ast.op, ast.And) else "or"
            sub_conditions = [ast.unparse(val) for val in condition_ast.values]
            
            return {
                "has_short_circuit": True,
                "operator": op_type,
                "sub_conditions": sub_conditions,
                "evaluation_order": sub_conditions  # Left to right
            }
        
        return {"has_short_circuit": False}
    
    def _compute_path_state(self, path_nodes: List[str], node_map: Dict) -> Dict:
        """Compute variable state along a path"""
        state = {}
        for node_id in path_nodes:
            node = node_map.get(node_id)
            if node and node["type"] == "assign":
                for var in node.get("variables", []):
                    if var not in state:
                        state[var] = []
                    state[var].append({
                        "node": node_id,
                        "value": node.get("value")
                    })
        return state
    
    def _analyze_return_structure(self, return_value: str) -> Dict[str, Any]:
        """Analyze return value structure for validation (GAP 2)"""
        try:
            # Try to parse as AST to understand structure
            parsed = ast.parse(f"x = {return_value}", mode="exec")
            value_node = parsed.body[0].value
            
            if isinstance(value_node, ast.Dict):
                return {
                    "type": "dict",
                    "keys": [ast.unparse(k) if k else None for k in value_node.keys],
                    "validation": "Check key presence and types",
                    "expected_keys": [ast.unparse(k) for k in value_node.keys if k]
                }
            elif isinstance(value_node, ast.List) or isinstance(value_node, ast.Tuple):
                return {
                    "type": "list" if isinstance(value_node, ast.List) else "tuple",
                    "length": len(value_node.elts),
                    "validation": "Check length and element types"
                }
            elif isinstance(value_node, ast.Constant):
                return {
                    "type": type(value_node.value).__name__,
                    "value": value_node.value,
                    "validation": "Exact value match"
                }
            else:
                return {
                    "type": "complex",
                    "expression": return_value,
                    "validation": "Evaluate and compare"
                }
        except:
            return {
                "type": "unknown",
                "raw": return_value,
                "validation": "Compare as string or evaluate"
            }
    
    def _generate_test_guidance(self, cfg_nodes: List[Dict], paths: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive guidance for LLM test generation"""
        
        # Extract literals (boundary values)
        literals = self._extract_literals()
        
        # Identify loops with real unrolling guidance
        loops = [n for n in cfg_nodes if n["type"] == "loop"]
        loop_guidance = []
        for loop in loops:
            loop_guidance.append({
                "loop": loop["code"],
                "variable": loop.get("loop_variable"),
                "iterable": loop.get("iterable"),
                "test_scenarios": [
                    {
                        "iterations": 0, 
                        "description": "Empty iterable",
                        "constraint": f"len({loop.get('iterable')}) == 0",
                        "example_input": "[]"
                    },
                    {
                        "iterations": 1, 
                        "description": "Single element",
                        "constraint": f"len({loop.get('iterable')}) == 1",
                        "example_input": "[x]"
                    },
                    {
                        "iterations": 3, 
                        "description": "Multiple elements",
                        "constraint": f"len({loop.get('iterable')}) > 1",
                        "example_input": "[x, y, z]"
                    }
                ]
            })
        
        # Identify exception paths with triggering constraints
        exception_paths = [p for p in paths if p["outcome"]["type"] == "raise"]
        exception_details = []
        for ep in exception_paths:
            exception_details.append({
                "exception": ep["outcome"]["value"],
                "path_id": ep["path_id"],
                "triggering_constraints": ep["constraints"],
                "how_to_trigger": self._explain_exception_trigger(ep["constraint_details"])
            })
        
        # Get variable mutations per path
        assignments = [n for n in cfg_nodes if n["type"] == "assign"]
        mutated_vars = {}
        for assign in assignments:
            for var in assign.get("variables", []):
                if var not in mutated_vars:
                    mutated_vars[var] = []
                mutated_vars[var].append({
                    "node": assign["id"],
                    "value": assign["value"],
                    "affects_paths": self._find_paths_with_node(assign["id"], paths)
                })
        
        # Short-circuit analysis with detailed guidance (if enabled)
        short_circuits = []
        if self.config.enable_short_circuit_detection:
            for node in cfg_nodes:
                if node.get("type") == "decision":
                    cond = node.get("condition", "")
                    if " and " in cond:
                        parts = cond.split(" and ")
                        short_circuits.append({
                            "expression": cond,
                            "operator": "and",
                            "components": parts,
                            "test_cases": [
                                f"Test {parts[0]} == False (short-circuits, {parts[1]} not evaluated)",
                                f"Test {parts[0]} == True and {parts[1]} == False",
                                f"Test both True"
                            ]
                        })
                    elif " or " in cond:
                        parts = cond.split(" or ")
                        short_circuits.append({
                            "expression": cond,
                            "operator": "or",
                            "components": parts,
                            "test_cases": [
                                f"Test {parts[0]} == True (short-circuits, {parts[1]} not evaluated)",
                                f"Test {parts[0]} == False and {parts[1]} == True",
                                f"Test both False"
                            ]
                        })
        
        # Path-specific variable states
        path_states = []
        for path in paths:
            state = path.get("state", {})
            if state:
                path_states.append({
                    "path_id": path["path_id"],
                    "variables": state,
                    "description": self._describe_path_state(state)
                })
        
        return {
            "total_paths": len(paths),
            "boundary_values": literals,
            "loops": loop_guidance,
            "exception_details": exception_details,
            "variable_mutations": mutated_vars,
            "short_circuits": short_circuits,
            "path_states": path_states,
            "coverage_targets": {
                "statement_coverage": len([n for n in cfg_nodes if n["type"] not in ["entry", "merge"]]),
                "branch_coverage": len([n for n in cfg_nodes if n["type"] == "decision"]) * 2,
                "path_coverage": len(paths),
                "loop_coverage": len(loops) * 3  # 0, 1, N iterations
            }
        }
    
    def _explain_exception_trigger(self, constraints: List[Dict]) -> str:
        """Explain how to trigger an exception"""
        if not constraints:
            return "Any input reaching this path"
        
        explanations = []
        for c in constraints:
            if c["branch"] == "true":
                explanations.append(f"Make {c['expr']} true")
            else:
                explanations.append(f"Make {c['expr']} false")
        
        return " AND ".join(explanations)
    
    def _find_paths_with_node(self, node_id: str, paths: List[Dict]) -> List[str]:
        """Find which paths contain a node"""
        return [p["path_id"] for p in paths if node_id in p["nodes"]]
    
    def _describe_path_state(self, state: Dict) -> str:
        """Describe the variable state in a path"""
        descriptions = []
        for var, assignments in state.items():
            if len(assignments) == 1:
                descriptions.append(f"{var} = {assignments[0]['value']}")
            else:
                descriptions.append(f"{var} mutated {len(assignments)} times")
        return ", ".join(descriptions) if descriptions else "No mutations"
    
    def _extract_literals(self) -> Dict[str, List]:
        """Extract literal values for boundary testing"""
        literals = {"numbers": [], "strings": [], "booleans": []}
        
        for node in ast.walk(self.func_node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    literals["numbers"].append(node.value)
                elif self.config.extract_string_literals and isinstance(node.value, str) and len(node.value) < 50:
                    literals["strings"].append(node.value)
                elif isinstance(node.value, bool):
                    literals["booleans"].append(node.value)
        
        # Deduplicate and sort
        literals["numbers"] = sorted(list(set(literals["numbers"])))
        literals["strings"] = list(set(literals["strings"]))
        literals["booleans"] = list(set(literals["booleans"]))
        
        return literals


# ========================================
# LLM PROMPT GENERATOR
# ========================================
class LLMPromptGenerator:
    """Generate optimized prompts for LLM test generation"""
    
    @staticmethod
    def generate_prompt(payload: LLMTestGenerationPayload) -> str:
        """Create a complete prompt for LLM test generation"""
        
        data = json.loads(payload.to_json())
        
        prompt = f"""# Test Generation Task

## Function to Test
```python
{data['signature']['name']}({', '.join([f"{p['name']}: {p['type']}" for p in data['signature']['parameters']])}) -> {data['signature']['return_type']}
```

{f"**Purpose**: {data['signature']['docstring']}" if data['signature']['docstring'] else ""}

## Execution Paths
Total paths to cover: {data['guidance']['total_paths']}

"""
        
        for i, path in enumerate(data['paths'], 1):
            prompt += f"### Path {i}\n"
            prompt += f"- **Constraints**: {', '.join(path['constraints']) if path['constraints'] else 'None (straight-line execution)'}\n"
            prompt += f"- **Outcome**: {path['outcome']['type']} {path['outcome']['value']}\n\n"
        
        prompt += "## Test Guidance\n\n"
        
        # Boundary values
        if data['guidance']['boundary_values']['numbers']:
            prompt += f"**Boundary Values**: Test with {data['guidance']['boundary_values']['numbers']}\n\n"
        
        # Loops
        if data['guidance']['loops']:
            prompt += "**Loop Testing**:\n"
            for loop in data['guidance']['loops']:
                prompt += f"- Loop: `{loop['loop']}`\n"
                for scenario in loop['test_scenarios']:
                    prompt += f"  - {scenario['iterations']} iterations: {scenario['description']}\n"
                    prompt += f"    Constraint: {scenario['constraint']}\n"
                    prompt += f"    Example: {scenario['example_input']}\n"
            prompt += "\n"
        
        # Exceptions
        if data['guidance']['exception_details']:
            prompt += "**Exception Testing**:\n"
            for exc in data['guidance']['exception_details']:
                prompt += f"- Exception: `{exc['exception']}`\n"
                prompt += f"  Path: {exc['path_id']}\n"
                prompt += f"  How to trigger: {exc['how_to_trigger']}\n"
                prompt += f"  Constraints: {', '.join(exc['triggering_constraints'])}\n"
            prompt += "\n"
        
        # Short circuits
        if data['guidance']['short_circuits']:
            prompt += "**Short-Circuit Testing**:\n"
            for sc in data['guidance']['short_circuits']:
                prompt += f"- `{sc['expression']}` ({sc['operator']}):" + "\n"
                for tc in sc['test_cases']:
                    prompt += f"  - {tc}\n"
            prompt += "\n"
        
        # Path states
        if data['guidance'].get('path_states'):
            prompt += "**Path-Specific Variable States**:\n"
            for ps in data['guidance']['path_states']:
                prompt += f"- {ps['path_id']}: {ps['description']}\n"
            prompt += "\n"
        
        prompt += """## Required Output Format
Generate test cases in this JSON format:

```json
{
  "test_cases": [
    {
      "test_id": "test_1",
      "description": "Test description",
      "path_id": "path_1",
      "input": {
        "param1": value1,
        "param2": value2
      },
      "expected_output": expected_value,
      "expected_exception": null
    }
  ]
}
```

## Requirements
1. Generate at least one test per execution path
2. Include boundary value tests
3. Test all loop iteration scenarios (0, 1, multiple)
4. Test exception paths with inputs that trigger them
5. Test short-circuit boolean expressions
6. Ensure input types match function signature

Generate comprehensive test cases now:
"""
        
        return prompt


# ========================================
# MAIN ENTRY POINT
# ========================================
def generate_llm_input(source_code: str, function_name: str, config: Optional[AnalyzerConfig] = None) -> Dict[str, Any]:
    """
    Main function: Takes source code, returns LLM-ready JSON
    
    Args:
        source_code: Python source code containing the function
        function_name: Name of the function to test
        config: Optional AnalyzerConfig for customizing behavior
    
    Returns:
        Dictionary with 'payload' (JSON for LLM) and 'prompt' (formatted prompt)
    
    Raises:
        ValueError: If source code is invalid or function not found
    """
    try:
        analyzer = LLMReadyAnalyzer(source_code, function_name, config)
        payload = analyzer.analyze_for_llm()
        prompt = LLMPromptGenerator.generate_prompt(payload)
        
        return {
            "payload": json.loads(payload.to_json()),
            "prompt": prompt,
            "raw_json": payload.to_json()
        }
    except ValueError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}", exc_info=True)
        raise ValueError(f"Analysis failed: {str(e)}")


# ========================================
# DEMO
# ========================================
if __name__ == "__main__":
    # Example 1: Function with loops and conditions
    test_code_1 = """
def calculate_statistics(numbers: list, threshold: int) -> dict:
    '''Calculate sum, count, and average for numbers above threshold'''
    if not numbers:
        raise ValueError("List cannot be empty")
    
    total = 0
    count = 0
    
    for num in numbers:
        if num > threshold:
            total = total + num
            count = count + 1
    
    if count > 0:
        average = total / count
    else:
        average = 0
    
    return {"sum": total, "count": count, "average": average}
"""
    
    # Example 2: Payment processing
    test_code_2 = """
def process_payment(amount: float, balance: float, discount: int) -> dict:
    '''Process payment with optional discount'''
    if amount <= 0 or balance < 0:
        return {"status": "invalid"}
    
    final_amount = amount
    if discount > 0 and discount <= 100:
        final_amount = amount * (1 - discount / 100)
    
    if balance >= final_amount:
        balance = balance - final_amount
        return {"status": "success", "remaining": balance}
    else:
        return {"status": "insufficient_funds"}
"""
    
    print("="*80)
    print("LLM-READY ANALYZER - DEMO")
    print("="*80)
    
    for i, (code, func_name) in enumerate([
        (test_code_1, "calculate_statistics"),
        (test_code_2, "process_payment")
    ], 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}: {func_name}")
        print("="*80)
        
        result = generate_llm_input(code, func_name)
        
        print("\n[1] JSON PAYLOAD (for programmatic LLM API calls):")
        print("-" * 80)
        print(json.dumps(result['payload'], indent=2))
        
        print(f"\n[2] FORMATTED PROMPT (for chat-based LLMs):")
        print("-" * 80)
        print(result['prompt'])
        
        print("\n" + "="*80)
        print(f"Ready to send to LLM API")
        print(f"   - Paths identified: {len(result['payload']['paths'])}")
        print(f"   - Guidance items: {len(result['payload']['guidance'])}")
        print("="*80)
