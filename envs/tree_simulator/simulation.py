#! usr/bin/python
import random
from collections import defaultdict

class DirNode:
    def __init__(self, name):
        self.name = name
        self.children = {}          # name → DirNode

    def mkdir_p(self, path_parts):
        if not path_parts:
            return
        head, *tail = path_parts
        if head not in self.children:
            self.children[head] = DirNode(head)
        self.children[head].mkdir_p(tail)

    # ASCII “tree” like the real command
    def render(self, prefix=""):
        lines = [prefix[:-3] + prefix[-3:].replace("├", "└") if prefix else "."]
        items = sorted(self.children.values(), key=lambda n: n.name)
        for i, child in enumerate(items):
            connector = "└── " if i == len(items)-1 else "├── "
            spacer    = "    " if i == len(items)-1 else "│   "
            lines.append(prefix + connector + child.name)
            lines.extend(child.render(prefix + spacer)[1:])  # skip child's own dot
        return lines

class ShellSimulator:
    def __init__(self):
        self.root = DirNode(".")
        self.cwd  = self.root
        self.cwd_path = []

    def _resolve_cwd(self):
        node = self.root
        for part in self.cwd_path:
            node = node.children[part]
        self.cwd = node

    def run(self, commands):
        visible_ls = []
        for cmd in commands:
            if cmd.startswith("mkdir -p "):
                path = cmd.split(None, 2)[2].strip("/")
                parts = path.split("/")
                self.cwd.mkdir_p(parts)
            elif cmd == "cd ..":
                if self.cwd_path:
                    self.cwd_path.pop()
                    self._resolve_cwd()
            elif cmd == "cd ~":
                self.cwd_path.clear()
                self.cwd = self.root
            elif cmd == "ls":
                visible_ls = sorted(self.cwd.children)
            elif cmd.startswith("cd "):
                dst = cmd.split()[1]
                if dst in visible_ls:
                    self.cwd_path.append(dst)
                    self._resolve_cwd()
                else:
                    pass
                    # raise ValueError(f"cd: {dst} not found after ls")
            else:
                raise ValueError(f"Unsupported command: {cmd}")

    def tree_output(self):
        return "\n".join(self.root.render())

def generate_random_session(max_depth=3, max_ops=25, seed=None):
    rng = random.Random(seed)
    sim = ShellSimulator()
    cmds = []
    for _ in range(max_ops):
        op = rng.choice(["mkdir", "cdback", "cdroot", "ls", "cdin"])
        if op == "mkdir":
            depth = rng.randint(1, max_depth)
            parts = [f"dir{rng.randint(1,9)}" for _ in range(depth)]
            cmds.append("mkdir -p " + "/".join(parts))
        elif op == "cdback":
            cmds.append("cd ..")
        elif op == "cdroot":
            cmds.append("cd ~")
        elif op == "ls":
            cmds.append("ls")
        elif op == "cdin":
            # placeholder; will be pruned if no ls yet
            cmds.append("cd placeholder")
    # Fix invalid 'cd placeholder' with actual names
    valid_cmds = []
    sim = ShellSimulator()
    for c in cmds:
        if c == "cd placeholder":
            choices = sorted(sim.cwd.children)
            if choices:
                c = "cd " + rng.choice(choices)
            else:
                continue  # skip if nothing to cd into
        valid_cmds.append(c)
        sim.run([c])  # update state incrementally
    return valid_cmds, sim.tree_output()

if __name__ == "__main__":
    transcript, answer = generate_random_session(seed=227)
    print("# Commands:")
    for line in transcript:
        print(line)
    print("\n# Expected tree:")
    print(answer)
