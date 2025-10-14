

class SAMNode:
    def __init__(self):
        self.next = {}       # transitions
        self.link = -1       # suffix link
        self.len = 0         # max length of substring ending here

class SuffixAutomaton:
    def __init__(self):
        self.nodes = [SAMNode()]
        self.last = 0

    def extend(self, c):
        p = self.last
        cur = len(self.nodes)
        node = SAMNode()
        node.len = self.nodes[p].len + 1
        self.nodes.append(node)

        while p >= 0 and c not in self.nodes[p].next:
            self.nodes[p].next[c] = cur
            p = self.nodes[p].link

        if p == -1:
            node.link = 0
        else:
            q = self.nodes[p].next[c]
            if self.nodes[p].len + 1 == self.nodes[q].len:
                node.link = q
            else:
                clone = len(self.nodes)
                cloned_node = SAMNode()
                cloned_node.next = self.nodes[q].next.copy()
                cloned_node.link = self.nodes[q].link
                cloned_node.len = self.nodes[p].len + 1
                self.nodes.append(cloned_node)

                while p >= 0 and self.nodes[p].next.get(c) == q:
                    self.nodes[p].next[c] = clone
                    p = self.nodes[p].link

                self.nodes[q].link = clone
                node.link = clone

        self.last = cur

def lz77_complexity_linear(seq):
    """
    Linear-time Lempel-Ziv 77 complexity using suffix automaton.
    Matches the original parsing exactly.
    """
    seq = ''.join(map(str, seq))
    n = len(seq)
    if n == 0:
        return 0

    sam = SuffixAutomaton()
    complexity = 0
    i = 0  # start of current phrase

    while i < n:
        length = 0
        node = 0
        j = i
        # extend as far as possible in the automaton
        while j < n and seq[j] in sam.nodes[node].next:
            node = sam.nodes[node].next[seq[j]]
            length += 1
            j += 1

        # new phrase found
        complexity += 1

        # insert all new characters of this phrase into the SAM
        for k in range(i, min(i + length + 1, n)):
            sam.extend(seq[k])

        i += length + 1  # move to next phrase
    return complexity

def lz_hybrid(seq, window_size = 0):
    """
    lz77-lz78 hybrid.
    Works for sequences of ints, chars, or strings.
    """
    seq = ''.join(map(str, seq))
    n = len(seq)
    if n == 0:
        return 0

    complexity = 0
    i = 0  # start index
    j = 1  # end index

    # Maintain a hash set of substrings we've seen 
    seen = set()
    while j <= n:
        substring = seq[i:j]
        fidx = max(i - window_size, 0) if window_size is not None else 0
        if substring in seen or substring in seq[fidx:j-1]:
            j += 1
        else:
            # New phrase found
            seen.add(substring)
            complexity += 1
            i = j
            j = i + 1
    return complexity

def lz77_py(seq):
    """ Lempel-Ziv 77 complexity \n
    Note: The input sequence is cast as string 
    and not tuple (for performance reasons).
    """

    if not isinstance(seq, str):
        seq = ''.join(map(str, seq))

    complexity, ind, inc = 1, 1, 0
    L = len(seq)
    while ind + inc < L:
        if seq[ind : ind + inc + 1] in seq[: ind + inc]:
            inc += 1
        else:
            complexity += 1
            ind += inc + 1
            inc = 0
    return complexity + 1 if inc != 0 else complexity

def lz78_py(seq):
    """
    Optimized Lempel-Ziv 78 complexity (number of phrases).
    Works for sequences of ints, chars, or strings.
    """
    
    if not isinstance(seq, str):
        seq = ''.join(map(str, seq))

    n = len(seq)
    if n == 0:
        return 0

    complexity = 0
    i = 0  # start index
    j = 1  # end index

    # Maintain a hash set of substrings we've seen
    seen = set()
    while j <= n:
        substring = seq[i:j]
        if substring in seen:
            j += 1
        else:
            # New phrase found
            seen.add(substring)
            complexity += 1
            i = j
            j = i + 1
    return complexity

def lz77_py_old(seq):
    """ Lempel-Ziv 77 complexity \n
    Note: The input sequence is cast as string 
    and not tuple (for performance reasons).
    """
    complexity, ind, inc = 1, 1, 0
    while ind + inc < len(seq):
        if seq[ind : ind + inc + 1] in seq[: ind + inc]:
            inc += 1
        else:
            complexity += 1
            ind += inc + 1
            inc = 0
    return complexity + 1 if inc != 0 else complexity

def lz78_py_old(seq):
    """ Lempel-Ziv 78 complexity \n
    Note: The input sequence is cast as string 
    and not tuple (for performance reasons).
    """
    sub_strings = set()
    ind, inc = 0, 1
    while ind + inc <= len(seq):
        sub_str = seq[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)
