def do_overlap(bbox1, bbox2):
    # 检查两个边界框是否重叠的函数
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    return not (x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1)
class CustomUnionFind:
    def __init__(self, compare_function=lambda x, y: x == y, hash_function=lambda x: x):
        self.parent = {}
        self.rank = {}
        self.compare_function = compare_function
        self.hash_function = hash_function

    def add(self, element):
        if element not in self.parent:
            self.parent[element] = element
            self.rank[element] = 0

    def find(self, element):
        hashed_element = self.hash_function(element)
        if hashed_element not in self.parent:
            return None  # Element not found
        return self._find(hashed_element)

    def initialize(self, elements):
        for e in elements:
            self.integrate(e)
    
    def compare(self, e1, e2):
        root1 = self.find(e1)
        root2 = self.find(e2)
        return root1 == root2

    def _find(self, element):
        if self.parent[element] != element:
            self.parent[element] = self._find(self.parent[element])
        return self.parent[element]

    def union(self, element1, element2):
        root1 = self._find(element1)
        root2 = self._find(element2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def integrate(self, element):
        hashed_element = self.hash_function(element)
        if hashed_element not in self.parent:
            self.add(hashed_element)
        for other_element in self.parent.keys():
            if self.compare_function(hashed_element, other_element):
                self.union(hashed_element, other_element)
                break
    
    def groups(self):
        groups = {}
        for i in self.parent.keys():
            root = self._find(i)
            if root in groups:
                groups[root].append(i)
            else:
                groups[root] = [i]

        return list(groups.values())
