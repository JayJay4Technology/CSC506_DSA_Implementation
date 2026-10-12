# Data Structures and Algorithms Learning Tool

This project is an interactive, menu-driven Python learning tool for exploring
data structures and algorithms. The implementation combines hand-written
structures, algorithm demonstrations, text-based visualizations, example use
cases, and runtime comparisons. It is intended for study and experimentation,
not as a production-ready replacement for Python's standard-library
collections.

The program helps answer two related questions:

1. **How is data organized?** A data structure defines how values are stored
   and related.
2. **How is a task performed?** An algorithm describes the steps used to
   search, sort, traverse, or otherwise process that data.

The best structure or algorithm depends on the operations a program performs
most often. For example, a list provides direct access by index, a stack
preserves last-in-first-out order, and a graph represents relationships among
vertices that need not form a hierarchy.

> **Interpret benchmark output as an experiment, not a guarantee.** Runtime
> measurements depend on the computer, Python build, background load, data
> shape, and other factors. Big-O describes how work tends to grow as input
> size increases; one timing does not prove an asymptotic complexity.

## Contents

- [Requirements and setup](#requirements-and-setup)
- [Starting and navigating the application](#starting-and-navigating-the-application)
- [Data structure foundations](#data-structure-foundations)
- [Linear data structures](#linear-data-structures)
- [Non-linear data structures](#non-linear-data-structures)
- [Searching and selection algorithms](#searching-and-selection-algorithms)
- [Sorting algorithms](#sorting-algorithms)
- [Hash tables, priority queues, and sets](#hash-tables-priority-queues-and-sets)
- [Complexity reference](#complexity-reference)
- [Using the implementation as a Python module](#using-the-implementation-as-a-python-module)
- [Benchmarking, memory estimates, and limitations](#benchmarking-memory-estimates-and-limitations)
- [Suggested learning path](#suggested-learning-path)

## Requirements and setup

- Python **3.10 or newer**. The source uses built-in generic annotations and
  union types introduced in modern Python versions.
- The standard library is sufficient for the menus, data structures, search,
  sorting, graph, and set features.
- `matplotlib` is optional. It is imported only when the sorting-chart
  generation function is called; the rest of the application can be run
  without it.

From the project directory, start the application with:

```bash
python data_structures_and_algorithms_implementation.py
```

To use the optional sorting charts, install `matplotlib` in the same Python
environment used to launch the program:

```bash
python -m pip install matplotlib
```

## Starting and navigating the application

The top-level menu displays the following categories:

| Choice | Category |
| --- | --- |
| `1` | Linear data structures: arrays, stacks, queues, and linked lists |
| `2` | Non-linear data structures: a binary-search-tree map explorer, then graph management |
| `3` | Sorting demonstrations, benchmark reports, and charts |
| `4` | Linear/binary search demonstrations, quickselect, and search benchmarks |
| `5` | Hash-table and priority-queue demonstrations |
| `6` | Custom-set demonstration |
| `0` | Exit the application |

Submenus have their own exit options, and their option numbers are not
necessarily the same as the main menu. Read the choices printed by the current
menu before entering a selection. The application stores demonstration data in
memory; it does not save menu changes to a file or database.

The non-linear menu runs the tree-map explorer first. Selecting `0` there
returns to the enclosing flow and opens the graph manager, which starts with a
sample graph containing vertices `A`, `B`, and `C`.

The linear-structure submenu has options for individual structures, a sample
test suite, use-case demonstrations, performance comparisons, and an
interactive test suite. Its option `9` exits the application rather than
returning to the top-level menu.

## Data structure foundations

### Linear and non-linear structures

Linear structures organize values in a sequence. Non-linear structures encode
branching or network relationships.

| Feature | Linear data structures | Non-linear data structures |
| --- | --- | --- |
| **Arrangement** | Values form an ordered sequence. | Values form hierarchies or networks. |
| **Relationships** | Usually described by neighboring positions or links. | A node may connect to multiple children or vertices. |
| **Traversal** | Often a sequential pass through the values. | May use specialized traversal strategies such as tree orders, BFS, or DFS. |
| **Examples in this project** | Array, stack, queue, deque, singly and doubly linked lists. | Binary search tree / map and graph. |

These categories describe organization, not a guarantee of speed or memory
efficiency. Actual costs depend on the operation and implementation. For
example, this project's queue uses a Python list, so removing its first element
requires shifting the remaining elements.

### Trees and graphs

A tree is a connected, acyclic structure with one unique path between every
pair of nodes. A rooted tree represents a hierarchy with one designated root.
A graph is a more general set of vertices and edges: it can be directed or
undirected, cyclic or acyclic, connected or disconnected.

| Feature | Tree | Graph |
| --- | --- | --- |
| **Hierarchy** | A rooted tree has a designated root and parent-child relationships. | No hierarchy is required; vertices are peers unless the application imposes roles. |
| **Cycles** | A tree is acyclic. | May contain cycles or be acyclic. |
| **Connectivity** | A non-empty tree is connected. | May be connected or disconnected, including isolated vertices. |
| **Paths** | Exactly one path exists between any two nodes. | There may be zero, one, or multiple paths between vertices. |
| **Edges** | A tree with `N` nodes has `N - 1` edges. | For a simple graph without self-loops, the maximum is `N(N - 1)` directed edges or `N(N - 1) / 2` undirected edges. |
| **Direction** | A rooted representation commonly points from parent to child; the underlying tree need not be directed. | May be directed or undirected. |
| **Common traversals** | Preorder, inorder (for binary trees), postorder, and level-order. | Breadth-first search (BFS) and depth-first search (DFS). |

## Linear data structures

The linear-structure submenu provides interactive operations, sample tests,
problem-solving examples, performance comparisons, and a further interactive
test suite. Several demos initialize structures with sample values so that
operations can be observed on a non-empty collection.

The use-case examples include balanced-bracket checking, processing print
jobs in queue order, palindrome checking, and a playlist implemented with a
doubly linked list.

### Array

`Array` is a small wrapper around Python's list. It supports appending a value,
reading by zero-based index, deleting by index, checking size, returning a list
copy, and estimating memory use.

- Indexed reads are `O(1)`.
- Appending is `O(1)` amortized.
- Deleting an arbitrary index is `O(n)` in the underlying list because later
  elements move to fill the gap.

The array demo is a learning abstraction; it does not allocate a low-level
fixed-size C-style array.

### Stack

`Stack` uses a Python list and follows **LIFO** (last in, first out). `push`
adds to the top, `pop` removes and returns the top item, and `peek` reads the
top without removing it. It also exposes `is_empty`, `size`, `clear`, and
`to_list`.

Push and pop at the end of a Python list are amortized `O(1)`. Calling `pop` or
`peek` on an empty stack raises `IndexError`.

### Queue

`Queue` follows **FIFO** (first in, first out). `enqueue` adds at the back,
`front` reads the first value, and `dequeue` removes and returns it. The class
also provides `is_empty`, `size`, `clear`, and `to_list`.

The implementation is backed by a Python list: enqueue is amortized `O(1)`,
but dequeue is `O(n)` because `pop(0)` shifts the remaining values. This
trade-off is intentionally visible in the demonstration.

### Deque

`Deque` is a double-ended queue with operations to add or remove at either end.
It is also implemented with a Python list, not `collections.deque`. Therefore
rear append/removal is typically `O(1)` amortized, while front insertion and
removal are `O(n)` because elements shift.

### Singly linked list

`SinglyLinkedList` stores values in nodes linked by a `next` reference. It
supports insertion at the head, search, deletion of the first matching value,
size, emptiness checks, forward display, and approximate memory reporting.

- Inserting at the head is `O(1)`.
- Searching and deleting by value are `O(n)` because the list may need to be
  scanned.
- Nodes are linked by references and do not provide direct indexed access.

### Doubly linked list

`DoublyLinkedList` stores `head` and `tail` references and each node has both
`next` and `prev` links. The demonstration covers insertion at both ends,
search, deletion, and forward/backward traversal.

- Insertion at the head or tail is `O(1)`.
- Search and deletion by value are `O(n)` because locating a value may require
  scanning.
- The extra backward link uses additional memory but allows traversal in both
  directions.

## Non-linear data structures

### Binary search tree and `TreeMap`

`BinarySearchTree` stores `(key, value)` pairs. Keys smaller than a node's key
are placed in its left subtree; larger keys are placed in its right subtree.
Inserting an existing key updates its value. The class supports search,
insertion, deletion, minimum/maximum lookup, inorder/preorder/postorder
traversals, height calculation, balance detection, BST-property validation,
and a sideways ASCII visualization.

After insertion and deletion, the implementation checks whether the height
difference between each node's left and right subtrees is at most one. If not,
it collects the inorder sequence and rebuilds a balanced tree. This is a
whole-tree rebuild strategy; it is **not an AVL implementation and does not
perform AVL rotations**.

`TreeMap` wraps the BST and offers mapping-style operations:

```python
tree_map[key] = value
value = tree_map[key]
del tree_map[key]
key in tree_map
```

It also provides `get`, `keys`, `values`, pre-order and post-order lists,
`find_min`, `find_max`, `visualize`, BST validation, and a formatted balance
status. Inorder keys are returned in ascending order.

The tree explorer demonstrates traversal orders, insertion/search/deletion,
minimum/maximum values, visualization, map-versus-list lookup timing, use-case
evaluation, and a 50-item test with mixed value types. Although values may have
different types, keys used together in a tree must be mutually orderable.

### Graph

`Graph` supports either directed or undirected edges and stores weighted
connections in two synchronized representations:

- **Adjacency list:** for each vertex, a list of neighboring vertices and edge
  weights. This is generally convenient for iterating over neighbors and for
  sparse graphs.
- **Adjacency matrix:** a two-dimensional matrix indexed by a maintained vertex
  order. This makes testing a specific edge a direct matrix lookup but uses
  `O(V²)` space.

The graph manager can add vertices, add or update weighted edges, remove
vertices and edges, display both representations, and run a comparison
benchmark. It asks whether the graph is directed when the manager opens;
adding an edge also creates its endpoint vertices if they are not present.

Available graph algorithms and behaviors:

- **BFS** visits reachable vertices level by level using a queue.
- **DFS** explores a path as deeply as possible before backtracking using a
  stack.
- **Dijkstra's algorithm** computes the minimum distance from a chosen source
  to every vertex. Unreachable vertices have infinite distance.
- **Traversal visualization** shows queue/stack changes as BFS or DFS proceeds.

The interactive Dijkstra option displays distances, not the actual sequence of
vertices in each shortest route. Dijkstra requires non-negative edge weights;
the `Graph` implementation does not validate weights, so use non-negative
weights when interpreting its results. A traversal starts at one vertex and
only returns vertices reachable from that start; it does not automatically
visit other disconnected components.

The graph-comparison option measures particular edge-check and neighbor-scan
operations on a generated undirected graph. Its observed timings are examples,
not universal rules about which representation will be faster.

## Searching and selection algorithms

### Linear search

`linear_search_visual(arr, target)` scans values in order and reports each
checked index. It works on unsorted input and returns the first matching index,
or `-1` when the target is absent.

- Best case: `O(1)` when the first item matches.
- Average and worst case: `O(n)`.
- No sorted-input requirement.

### Binary search

`binary_search_visual(arr, target)` shows the low, middle, and high positions
as it repeatedly halves the remaining search interval. It returns a matching
index or `-1`.

- Time: `O(log n)`.
- Input requirement: values must be sorted in ascending order.

The interactive search demo runs linear search on a predefined unsorted sample
and binary search on a sorted copy of that same sample.

### Quickselect

`quickselect(arr, k)` returns the 1-based `k`th-smallest value without sorting
the entire list. It uses a randomized pivot and partitions the input in place.
`k` must satisfy `1 <= k <= len(arr)`; invalid ranks raise `ValueError`.

- Average time: `O(n)`.
- Worst case: `O(n²)`.
- The input list is modified during partitioning; pass a copy if its original
  order must be retained.

The quickselect menu demo uses a generated sample and asks for the rank.

## Sorting algorithms

The module includes non-visual sorting functions that return new lists:

| Algorithm | Best case | Average / expected case | Worst case | Additional notes |
| --- | --- | --- | --- | --- |
| Bubble sort | `O(n)` | `O(n²)` | `O(n²)` | Repeatedly swaps adjacent out-of-order values; the best case uses an early-exit pass. |
| Selection sort | `O(n²)` | `O(n²)` | `O(n²)` | Selects and places the smallest remaining value each pass. |
| Insertion sort | `O(n)` | `O(n²)` | `O(n²)` | Best case is already-sorted input; reverse order is worst case. |
| Merge sort | `O(n log n)` | `O(n log n)` | `O(n log n)` | Bottom-up iterative implementation with `O(n)` workspace. |
| Quick sort | `O(n log n)` | `O(n log n)` expected | `O(n²)` | Median-of-three pivot selection and an explicit range stack improve practical behavior but do not remove the theoretical worst case. |

The sorting menu includes:

1. Dataset previews for a requested positive item count.
2. A performance benchmark across bubble, selection, insertion, merge, and
   quick sort.
3. A printed sorting report.
4. A step-by-step visualization for a selected sorting algorithm.
5. Optional performance charts.

The benchmark uses random, already-sorted, reverse-sorted, and partially-sorted
datasets at sizes `1,000`, `5,000`, and `10,000`. The partially-sorted dataset
starts sorted and applies swaps to a portion of its values. Each report or
chart request runs the benchmark again; benchmark results are not persisted
between menu choices. A 180-second threshold prints an advisory warning for a
slow algorithm/data combination but does not stop it.

### Step-by-step visualizations

The sorting visualizer accepts integers separated by commas or spaces and
offers bubble, selection, insertion, merge, and quick sort visualizations.
`bubble_sort_visual` is also callable directly: it verifies that the input is a
list of integers (booleans are rejected), returns a sorted copy, and leaves the
original unchanged. It reports each adjacent comparison and swap and exits
early after a pass with no swaps. The other visual sort functions print
algorithm-specific progress.

### Performance charts

Chart generation creates one PNG per dataset type under a `charts/` directory
relative to the current working directory. The chart function imports
`matplotlib.pyplot` on demand. Install `matplotlib` only if this feature is
needed.

## Hash tables, priority queues, and sets

### Hash table

`HashTable` maps hashable keys to values and resolves collisions using separate
chaining. It supports:

- `insert(key, value)` to add a key or replace its existing value.
- `search(key)` to return its value, or `None` when absent.
- `delete(key)` to remove a key and return whether it existed.
- `items()` to iterate over key-value pairs and a `size` property.

The default capacity is 17 buckets and the bucket count does not automatically
resize. Lookup, insertion, and deletion are `O(1)` on average with a good
distribution, but can degrade to `O(n)` when many keys collide in the same
bucket.

### Priority queue

`PriorityQueue` is a min-priority queue implemented as a binary heap. Smaller
numeric priorities are served first. If multiple items have equal priority,
the queue uses insertion order as a tie-breaker.

- `insert(value, priority)` inserts an item.
- `delete()` removes and returns the next `(value, priority)` pair, or `None`
  when empty.
- `search(value)` finds an item by value.
- `items()` returns the queue contents ordered by priority.

Insertion and removal take `O(log n)`; searching for an arbitrary value takes
`O(n)`.

The combined menu loads 100 sample records, then supports listing them, adding
or updating hash entries, searching/deleting keys, adding/finding priority
tasks, serving the next task, comparing hash lookup with linear search, and
reloading the sample data.

### Custom set

`CustomSet` uses a Python dictionary to store unique hashable elements. It
supports adding, removing, discarding without error, clearing, membership
checks, size, conversion to a list, and the standard set operations:

- **Union:** elements from either set.
- **Intersection:** elements present in both.
- **Difference:** elements in the first set but not the second.
- **Symmetric difference:** elements present in exactly one set.

Membership, add, and remove are `O(1)` on average. The set demo first displays
the four operations on sample sets, then provides an interactive set where
values can be added, removed, checked, or cleared. Values entered through that
menu are strings.

## Complexity reference

These costs describe the algorithms and implementations in this project under
typical assumptions. `n` is the number of values, `V` is the number of graph
vertices, `E` is the number of graph edges, and `h` is the tree height.

“Best” and “worst” describe input-dependent costs where those differ. Amortized
costs account for occasional expensive operations such as dynamic-array
resizing. For hash-backed operations, average-case bounds assume a reasonable
hash distribution.

| Operation | Best case | Average / expected case | Worst case | Conditions and implementation notes |
| --- | --- | --- | --- | --- |
| Array indexed access | `O(1)` | `O(1)` | `O(1)` | Python list index access. |
| Array append | `O(1)` | `O(1)` amortized | `O(n)` | A resize may copy the existing list. |
| Array deletion by index | `O(1)` | `O(n)` | `O(n)` | Deleting the final element is constant-time; deleting earlier elements shifts values. |
| Stack push | `O(1)` | `O(1)` amortized | `O(n)` | Uses Python list append; resize may copy the list. |
| Stack pop/peek | `O(1)` | `O(1)` | `O(1)` | Pop/peek raise `IndexError` on an empty stack. |
| Queue enqueue | `O(1)` | `O(1)` amortized | `O(n)` | Uses Python list append. |
| Queue dequeue | `O(1)` | `O(n)` | `O(n)` | `pop(0)` shifts all remaining items; best case is a queue with at most one item. |
| Deque rear add/remove | `O(1)` | `O(1)` amortized for add | `O(n)` for add, `O(1)` for remove | Rear add can resize the Python list. |
| Deque front add/remove | `O(1)` | `O(n)` | `O(n)` | List insertion/removal at index zero shifts values; on an empty/single-item deque the work is constant. |
| Linked-list insertion at head | `O(1)` | `O(1)` | `O(1)` | Creates and links one node. |
| Doubly linked-list insertion at tail | `O(1)` | `O(1)` | `O(1)` | The class tracks a tail reference. |
| Linked-list search/delete by value | `O(1)` | `O(n)` | `O(n)` | Best case finds/removes the head; worst case scans the list. |
| BST search | `O(1)` | `O(log n)` | `O(log n)` | The implementation restores its height-balance condition after successful mutations; assumes comparable keys. |
| BST min/max lookup | `O(1)` | `O(log n)` | `O(log n)` | Root is the requested extreme in the best case; otherwise follow a branch. |
| BST traversal / height / balance check | `O(n)` | `O(n)` | `O(n)` | Visits the tree nodes. |
| BST insertion/deletion | `O(n)` | `O(n)` | `O(n)` | The implementation performs a full-tree balance check after mutation and may rebuild the tree. |
| Graph BFS or DFS | `O(1)` | `O(Vr + Er)` | `O(V + E)` | `Vr` and `Er` are vertices and edges reachable from the start; an isolated start is the best case. |
| Adjacency-matrix edge check | `O(1)` | `O(1)` | `O(1)` | Assumes vertex indices are already known. |
| Adjacency-list edge check | `O(1)` | `O(degree(v))` | `O(degree(v))` | Checks the first neighbor for best case; may scan the full neighbor list. |
| Dijkstra with heap | `O(V)` | `O((V + E) log V)` | `O((V + E) log V)` | For a valid source, initialization visits all vertices; valid shortest distances require non-negative weights. |
| Hash lookup/insert/delete | `O(1)` | `O(1)` expected | `O(n)` | Worst case occurs when many keys share a collision chain. |
| Priority-queue insert | `O(1)` | `O(log n)` | `O(log n)` | Heap may need to sift an inserted value to the root. |
| Priority-queue delete | `O(1)` | `O(log n)` | `O(log n)` | Removing the only item is constant-time; otherwise sift down. |
| Priority-queue search | `O(1)` | `O(n)` | `O(n)` | Scans heap entries; first item may match immediately. |
| Set membership/add/remove | `O(1)` | `O(1)` expected | `O(n)` | Dictionary-backed; average assumes normal hash behavior. |
| Linear search | `O(1)` | `O(n)` | `O(n)` | Best case matches the first element; works on unsorted data. |
| Binary search | `O(1)` | `O(log n)` | `O(log n)` | Best case matches the initial midpoint; input must be sorted. |
| Bubble sort | `O(n)` | `O(n²)` | `O(n²)` | Best case is already sorted because the implementation exits after a no-swap pass. |
| Selection sort | `O(n²)` | `O(n²)` | `O(n²)` | Scans the remaining suffix on every pass regardless of input order. |
| Insertion sort | `O(n)` | `O(n²)` | `O(n²)` | Best case is already sorted; worst case is reverse order. |
| Merge sort | `O(n log n)` | `O(n log n)` | `O(n log n)` | Bottom-up implementation uses an `O(n)` auxiliary workspace. |
| Quick sort | `O(n log n)` | `O(n log n)` expected | `O(n²)` | Median-of-three pivot selection helps common ordered inputs but does not remove the theoretical worst case. |
| Quickselect | `O(n)` | `O(n)` expected | `O(n²)` | Randomized pivots; worst-case partition sequences remain possible. |

## Using the implementation as a Python module

Importing the module defines its classes and functions without starting the
top-level interactive menu. For example:

```python
from data_structures_and_algorithms_implementation import (
    BinarySearchTree,
    CustomSet,
    DoublyLinkedList,
    Graph,
    HashTable,
    PriorityQueue,
    Stack,
    TreeMap,
    bubble_sort_visual,
    quickselect,
)

stack = Stack()
stack.push("first")
stack.push("second")
assert stack.pop() == "second"

linked = DoublyLinkedList()
linked.insert_at_head("middle")
linked.insert_at_tail("last")
assert linked.display() == ["middle", "last"]

tree = BinarySearchTree()
for key in (8, 3, 10):
    tree.insert(key, str(key))
assert [node.key for node in tree.traverse_in_order()] == [3, 8, 10]

tree_map = TreeMap()
tree_map[12] = "twelve"
assert tree_map.get(12) == "twelve"

graph = Graph(is_directed=False)
graph.add_edge("home", "station", weight=4)
graph.add_edge("station", "office", weight=2)
assert graph.dijkstra("home")["office"] == 6.0

table = HashTable[str, str]()
table.insert("course", "CSC506")
assert table.search("course") == "CSC506"

queue = PriorityQueue[str]()
queue.insert("normal task", priority=5)
queue.insert("urgent task", priority=1)
assert queue.delete() == ("urgent task", 1)

values = [5, 1, 4, 2]
sorted_values = bubble_sort_visual(values, silent=True)
assert sorted_values == [1, 2, 4, 5]
assert values == [5, 1, 4, 2]
assert quickselect([7, 2, 9, 1], 2) == 2

custom = CustomSet(["red", "blue"])
custom.add("green")
assert "green" in custom
```

The module also exposes the interactive functions `linear_data_structure_implementation`,
`non_linear_data_structure_implementation`, `run_sorting_demo`,
`run_searching_demo`, `run_hash_table_and_priority_queue_demo`, and
`run_set_demo`. Call them when an interactive terminal is available.

## Benchmarking, memory estimates, and limitations

- Timing is machine- and input-dependent. Use benchmark results to compare
  trends under the same conditions, not as universal performance claims.
- The sorting benchmark includes quadratic algorithms and can take a long
  time, particularly on large or adversarial inputs. Its 180-second threshold
  is advisory and does not cancel a run.
- The searching benchmark includes arrays up to one million values, performs
  five timed runs per scenario, and compares linear search, binary search, and
  quickselect. Its “Target Found” and “Target Missing” labels apply to the
  linear and binary searches. Quickselect does not use the target: it measures
  the same middle rank (`size // 2`) in both scenarios, so those two Quickselect
  readings are redundant rather than distinct target cases. Running this
  benchmark is more expensive than a small unit test.
- The tree rebalancing operation rebuilds the tree after detecting imbalance;
  it is not an incremental AVL or red-black tree implementation.
- `Graph.add_edge` does not validate endpoint labels or edge weights. In
  particular, an empty endpoint can be rejected by `add_node` but still be
  stored as an edge by `add_edge`, leaving adjacency data inconsistent with
  the vertex set. Negative, infinite, and NaN weights are also accepted even
  though Dijkstra requires finite, non-negative weights for meaningful
  shortest-distance results.
- The graph manager shows Dijkstra distance values rather than reconstructed
  shortest routes.
- The fixed-capacity hash table does not resize automatically.
- The queue and deque use Python lists, so front removals/insertions are
  `O(n)`.
- Memory-reporting methods use `sys.getsizeof` and manually include some
  referenced objects. These are approximate comparisons, not a full measure of
  process memory, allocator overhead, or all indirectly referenced objects.
- Interactive state is temporary. Exiting a demo does not persist its data.

## Suggested learning path

1. Begin with the array, stack, queue, and linked-list menus; compare their
   ordering rules and operation costs.
2. Use the linear and binary search visualizer with the same target. Notice
   that binary search requires sorted data.
3. Enter a short list such as `5, 1, 4, 2` into the sorting visualizer. Compare
   the repeated comparisons in bubble sort with the steps used by merge or
   quick sort.
4. In the tree explorer, compare insertion orders such as
   `8, 4, 12, 2, 6, 10, 14` and `1, 2, 3, 4, 5, 6, 7`. Inspect traversal order,
   height, validation, and balance status.
5. In graph management, create a disconnected vertex and compare the vertices
   visited from different starting points. Try both directed and undirected
   configurations.
6. Compare graph BFS/DFS traversal with Dijkstra distances. BFS explores by
   number of edges; Dijkstra minimizes total edge weight when weights are
   non-negative.
7. Run benchmarks only after exploring the smaller demonstrations, and keep
   theoretical complexity separate from observed elapsed time.
