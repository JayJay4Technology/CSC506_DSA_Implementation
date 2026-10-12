"""Interactive data-structure learning tool.

This module demonstrates linear data structures and non-linear structures
(binary search trees and weighted graphs) while teaching algorithmic complexity
concepts such as time and space usage.

The program is intentionally interactive: each menu option lets the user perform a
sample operation, predict its auxiliary space complexity, and compare that
prediction to measured execution time and memory usage.
"""

from __future__ import annotations

import heapq
import random
import sys
import time
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

# ================================================================
# LINEAR DATA STRUCTURES IMPLEMENTATION
# ================================================================


class Array:
    """A simple wrapper around Python's built-in list to illustrate array behavior.

    The class is a thin abstraction that allows the user to see how arrays behave
    in terms of time and space complexity.
    """

    def __init__(self, initial_data=None):
        """Initialize the array with optional initial data."""
        self._data = list(initial_data) if initial_data else []

    def append(self, item):
        """Add an item to the end of the array."""
        self._data.append(item)

    def get(self, index):
        """Retrieve an item by its index."""
        return self._data[index]

    def size(self):
        """Return the number of items in the array."""
        return len(self._data)

    def to_list(self):
        """Return a copy of the array contents as a list."""
        return list(self._data)

    def delete(self, index):
        """Remove an item at a specific index."""
        del self._data[index]

    def get_memory_bytes(self):
        """Estimate the memory footprint used by this array and its contents."""
        visited = {id(self), id(self._data)}
        total_bytes = sys.getsizeof(self) + sys.getsizeof(self._data)

        for item in self._data:
            if id(item) not in visited:
                visited.add(id(item))
                total_bytes += sys.getsizeof(item)

        return total_bytes


class Node:
    """Represents a linked-list node with forward and backward links.

    Singly linked lists use ``next``; doubly linked lists use both ``next`` and
    ``prev``.
    """

    def __init__(self, data):
        """Create a node with the given payload value."""
        self.data = data
        self.next = None
        self.prev = None


class Stack:
    """
    A Last-In, First-Out (LIFO) Stack implementation using Python lists.
    
    Elements are appended to the end of the internal storage and popped from 
    the same end, preserving a strict historical nesting order.
    """
    def __init__(self) -> None:
        """Initializes an empty stack structure."""
        self._items: List[Any] = []

    def push(self, item: Any) -> None:
        """
        Pushes an element onto the top of the stack.
        
        Time Complexity: O(1) amortized.
        """
        self._items.append(item)

    def pop(self) -> Any:
        """
        Removes and returns the top element of the stack.

        Time Complexity: O(1)
        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()

    def peek(self) -> Any:
        """
        Returns the top element of the stack without removing it.

        Time Complexity: O(1)
        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        """
        Checks whether the stack contains no elements.

        Time Complexity: O(1)
        """
        return len(self._items) == 0

    def size(self) -> int:
        """Return the number of items in the stack."""
        return len(self._items)

    def clear(self) -> None:
        """Remove all items from the stack."""
        self._items.clear()

    def to_list(self) -> List[Any]:
        """Return a copy of the stack contents in forward order."""
        return list(self._items)

    def get_memory_bytes(self):
        """Estimate the memory footprint used by this stack and its contents.

        The method tracks objects already visited to prevent double-counting shared
        references. This is useful for comparing memory behavior during different
        operations.
        """
        visited = {id(self), id(self._items)}
        total_bytes = sys.getsizeof(self) + sys.getsizeof(self._items)

        for item in self._items:
            if id(item) not in visited:
                visited.add(id(item))
                total_bytes += sys.getsizeof(item)

        return total_bytes


class Queue:
    """
    A First-In, First-Out (FIFO) Queue implementation using Python lists.
    
    Elements enter at the rear (enqueue) and exit from the front (dequeue).
    """

    def __init__(self) -> None:
        """Initializes an empty queue structure."""
        self._items: List[Any] = []

    def enqueue(self, item: Any) -> None:
        """
        Adds an element to the back of the queue.
        
        Time Complexity: O(1) amortized.
        """
        self._items.append(item)

    def dequeue(self) -> Any:
        """
        Removes and returns the front element of the queue.

        Time Complexity: O(n) due to element shifting in standard Python lists.
        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self._items.pop(0)

    def front(self) -> Any:
        """
        Returns the item at the front of the queue without removing it.

        Time Complexity: O(1)
        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0]

    def is_empty(self) -> bool:
        """
        Checks whether the queue contains no elements.

        Time Complexity: O(1)
        """
        return len(self._items) == 0

    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self._items)

    def clear(self) -> None:
        """Remove all items from the queue."""
        self._items.clear()

    def to_list(self) -> List[Any]:
        """Return a copy of the queue contents in current order."""
        return list(self._items)

    def get_memory_bytes(self):
        """Estimate the memory footprint used by this queue and its contents."""
        visited = {id(self), id(self._items)}
        total_bytes = sys.getsizeof(self) + sys.getsizeof(self._items)

        for item in self._items:
            if id(item) not in visited:
                visited.add(id(item))
                total_bytes += sys.getsizeof(item)

        return total_bytes

class Deque:
    """
    A Double-Ended Queue (Deque) implementation using Python lists.
    
    Allows elements to be added or removed from both the front and rear bounds.
    """

    def __init__(self) -> None:
        """Initializes an empty double-ended queue."""
        self.items: List[Any] = []

    def add_front(self, item: Any) -> None:
        """
        Inserts an element at the front of the deque.

        Time Complexity: O(n) due to shifting subsequent items in memory.
        """
        self.items.insert(0, item)

    def add_rear(self, item: Any) -> None:
        """
        Appends an element to the back of the deque.

        Time Complexity: O(1) amortized.
        """
        self.items.append(item)

    def remove_front(self) -> Any:
        """
        Removes and returns the frontmost item of the deque.

        Time Complexity: O(n) due to shifting remaining elements down.
        Raises:
            IndexError: If the deque is empty.
        """
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.pop(0)

    def remove_rear(self) -> Any:
        """
        Removes and returns the rearmost item of the deque.

        Time Complexity: O(1)
        Raises:
            IndexError: If the deque is empty.
        """
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.pop()

    def is_empty(self) -> bool:
        """
        Checks whether the deque contains no elements.

        Time Complexity: O(1)
        """
        return len(self.items) == 0

    def size(self) -> int:
        """Return the number of items in the deque."""
        return len(self.items)

    def clear(self) -> None:
        """Remove all items from the deque."""
        self.items.clear()

    def to_list(self) -> List[Any]:
        """Return a copy of the deque contents in current order."""
        return list(self.items)

    def get_memory_bytes(self):
        """Estimate the memory footprint used by this deque and its contents."""
        visited = {id(self), id(self.items)}
        total_bytes = sys.getsizeof(self) + sys.getsizeof(self.items)

        for item in self.items:
            if id(item) not in visited:
                visited.add(id(item))
                total_bytes += sys.getsizeof(item)

        return total_bytes

class SinglyLinkedList:
    """A singly linked list with insertion at the head.

    A linked list stores data in nodes that point to the next node. This allows
    efficient insertion at the front but requires traversal to search or delete an
    item somewhere in the middle.
    """

    def __init__(self):
        """Create an empty linked list."""
        self.head = None
        self._size = 0

    def insert_at_head(self, data):
        """Insert a new node at the beginning of the list.

        The new node becomes the head, and the old head becomes its successor.
        Time complexity: O(1).
        """
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1


    def delete(self, target_data):
        """Delete the first node whose data matches target_data.

        Returns True if a matching node is removed; otherwise returns False.
        Time complexity: O(n) because the list must be traversed.
        """
        current = self.head
        previous = None

        while current:
            if current.data == target_data:
                if previous:
                    previous.next = current.next
                else:
                    self.head = current.next
                self._size -= 1
                return True
            previous = current
            current = current.next

        return False

    def search(self, target_data):
        """Search for a value in the linked list.

        Returns True if the item is found, otherwise False. This is a linear scan,
        so the time complexity is O(n).
        """
        current = self.head
        while current:
            if current.data == target_data:
                return True
            current = current.next
        return False

    def size(self):
        """Return the number of nodes in the list."""
        return self._size

    def is_empty(self):
        """Check if the linked list is empty."""
        return self._size == 0

    def display(self) -> List[Any]:
        """Return the list values in forward order for display."""
        values = []
        current = self.head
        while current:
            values.append(current.data)
            current = current.next
        return values

    def get_memory_bytes(self):
        """Estimate the memory used by the linked list structure.

        The traversal is iterative instead of recursive so it avoids Python's
        recursion limit even when the list is very large.
        """
        visited = {id(self)}
        total_bytes = sys.getsizeof(self)

        # Use an explicit stack to walk the list without recursion.
        todo_stack = [self.head] if self.head else []

        while todo_stack:
            curr_node = todo_stack.pop()
            if not curr_node:
                continue

            node_id = id(curr_node)
            if node_id not in visited:
                visited.add(node_id)
                total_bytes += sys.getsizeof(curr_node)

                # Count the stored payload value and any remaining node links.
                if id(curr_node.data) not in visited:
                    visited.add(id(curr_node.data))
                    total_bytes += sys.getsizeof(curr_node.data)

                if curr_node.next:
                    todo_stack.append(curr_node.next)

        return total_bytes



class DoublyLinkedList:
    """A doubly linked list with head and tail references.

    Each node points to both its predecessor and successor, allowing efficient
    insertion and deletion at both ends of the list.
    """

    def __init__(self):
        """Create an empty doubly linked list."""
        self.head = None
        self.tail = None
        self._size = 0

    def insert_at_head(self, data):
        """Insert a new node at the beginning of the list."""
        new_node = Node(data)
        new_node.next = self.head
        if self.head:
            self.head.prev = new_node
        else:
            self.tail = new_node  # First node becomes the tail as well.
        self.head = new_node
        self._size += 1

    def insert_at_tail(self, data):
        """Insert a new node at the end of the list."""
        new_node = Node(data)
        new_node.prev = self.tail
        if self.tail:
            self.tail.next = new_node
        else:
            self.head = new_node  # First node becomes the head as well.
        self.tail = new_node
        self._size += 1

    def delete(self, target_data):
        """Delete the first node whose data matches target_data."""
        current = self.head
        while current:
            if current.data == target_data:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next  # Update head if needed.
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev  # Update tail if needed.
                self._size -= 1
                return True
            current = current.next
        return False

    def search(self, target_data):
        """Search for a value in the doubly linked list."""
        current = self.head
        while current:
            if current.data == target_data:
                return True
            current = current.next
        return False

    def size(self):
        """Return the number of nodes in the list."""
        return self._size

    def display(self) -> List[Any]:
        """
        Return a list of all node data in forward order.
        This method is primarily for demonstration and debugging purposes.
        Time Complexity: O(n) where n is the number of nodes in the list.
        """
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result

    def clear(self):
        """Remove all nodes from the list."""
        self.head = None
        self.tail = None
        self._size = 0

    def get_memory_bytes(self):
        """Estimate the memory used by the doubly linked list structure."""
        visited = {id(self)}
        total_bytes = sys.getsizeof(self)

        # Use an explicit stack to walk the list without recursion.
        todo_stack = [self.head] if self.head else []

        while todo_stack:
            curr_node = todo_stack.pop()
            if not curr_node:
                continue

            node_id = id(curr_node)
            if node_id not in visited:
                visited.add(node_id)
                total_bytes += sys.getsizeof(curr_node)

                # Count the stored payload value and any remaining node links.
                if id(curr_node.data) not in visited:
                    visited.add(id(curr_node.data))
                    total_bytes += sys.getsizeof(curr_node.data)

                if curr_node.next:
                    todo_stack.append(curr_node.next)
                if curr_node.prev:
                    todo_stack.append(curr_node.prev)

        return total_bytes


# ================================================================
# COMPLEXITY BENCHMARK ENGINE
# ================================================================

def run_complexity_quiz(operation_name, correct_time_o, correct_space_o, struct_instance, action_callback):
    """Run an interactive complexity exercise for a chosen data-structure action.

    The function measures the selected action, compares it with the expected
    algorithmic complexity, and prints a short explanation of the result.
    """
    initial_bytes = struct_instance.get_memory_bytes()
    initial_size = struct_instance.size()

    print(
        f"\n🧠 [COMPLEXITY PREDICTOR] Current Dataset: {initial_size} "
        f"elements ({initial_bytes:,} bytes)"
    )
    print(f"Predict the complexities for running '{operation_name}':")
    print(f" -> Expected Time Complexity:  {correct_time_o}")
    print(
        " -> What is the auxiliary (additional) SPACE complexity of "
        "this operation?"
    )
    print(
        "    1. O(1) - Constant Space "
        "(Uses a fixed amount of extra memory)"
    )
    print(
        "    2. O(n) - Linear Space "
        "(Allocates extra memory proportional to data)"
    )

    user_guess = input("Your space prediction (1-2): ").strip()
    predicted_space = "O(1)" if user_guess == '1' else ("O(n)" if user_guess == '2' else "Unknown")

    # Record the exact operation duration so the learner can compare theory to
    # real-world runtime behavior.
    start_time = time.perf_counter()
    result = action_callback()
    end_time = time.perf_counter()

    final_bytes = struct_instance.get_memory_bytes()
    execution_duration_ns = (end_time - start_time) * 1_000_000_000
    byte_differential = final_bytes - initial_bytes

    print("\n" + "=" * 60)
    if predicted_space == correct_space_o:
        print(f"🎉 CORRECT SPACE PREDICTION! Operation uses {correct_space_o} auxiliary space.")
    else:
        print(f"❌ INCORRECT SPACE PREDICTION. You guessed {predicted_space}.")
        print(f"💡 Actual Auxiliary Space Complexity: {correct_space_o}")

    print("-" * 60)
    print(f"⏱  Execution Duration:  {execution_duration_ns:,.2f} nanoseconds")
    print(f"📊 Total Memory Now:     {final_bytes:,} bytes")
    print(f"📉 Memory Differential:  {byte_differential:+} bytes allocated during operation")
    print("=" * 60 + "\n")
    return result


# ================================================================
# START - LINEAR DATA STRUCTURE USER INTERFACE
# ================================================================

def run_array_demo():
    """Run the interactive array demonstration menu.

    The demo seeds the array with a large sample dataset and then lets the user
    test append and random access operations while observing their complexity
    characteristics.
    """
    array = Array()
    print("\n⚙️  Injecting 10,000 baseline items into the Array...")
    for i in range(10000):
        array.append(f"item_{i}")
    print("✅ Injection complete!")

    while True:
        print(f"--- ARRAY OPERATIONS (Elements: {array.size()} | RAM: {array.get_memory_bytes()} bytes) ---")
        print("1. Append New Item (Time: O(1) | Space: O(1))")
        print("2. Random Access by Index (Time: O(1) | Space: O(1))")
        print("3. Delete Item by Index (Time: O(n) | Space: O(1))")
        print("4. Search for Item by Value (Time: O(n) | Space: O(1))")
        print("5. Display Array Contents (Time: O(n) | Space: O(n))")
        print("0. Back to Main Menu")
        choice = input("Select an operation: ").strip()

        if choice == '1':
            val = input("Enter value to append: ").strip()
            if val:
                run_complexity_quiz("Array Append", "O(1)", "O(1)", array, lambda: array.append(val))
        elif choice == '2':
            index_str = input(f"Enter index to access (0 to {array.size() - 1}): ").strip()
            if index_str.isdigit():
                index = int(index_str)
                if 0 <= index < array.size():
                    accessed_value = run_complexity_quiz(
                        "Array Random Access", "O(1)", "O(1)", array,
                        lambda: array.get(index),
                    )
                    print(f"✅ Accessed Value at Index {index}: {accessed_value}")
                else:
                    print("\n⚠️ Index out of bounds!")
            else:
                print("\n⚠️ Invalid index input!")
        elif choice == '3':
            index_str = input(f"Enter index to delete (0 to {array.size() - 1}): ").strip()
            if index_str.isdigit():
                index = int(index_str)
                if 0 <= index < array.size():
                    deleted_value = run_complexity_quiz(
                        "Array Delete", "O(n)", "O(1)", array,
                        lambda: array.delete(index),
                    )
                    print(f"✅ Deleted Value at Index {index}: {deleted_value}")
                else:
                    print("\n⚠️ Index out of bounds!")
            else:
                print("\n⚠️ Invalid index input!")
        elif choice == '4':
            val = input("Enter value to search for: ").strip()
            if val:
                found = run_complexity_quiz(
                    "Array Search", "O(n)", "O(1)", array,
                    lambda: val in array.to_list(),
                )
                print(f"{'✅ Found' if found else '❌ Not found'}: {val}")
        elif choice == '5':
            contents = run_complexity_quiz(
                "Array Display", "O(n)", "O(n)", array,
                array.to_list,
            )
            print(f"Array Contents: {contents}")
        elif choice == '0':
            break


def run_stack_demo():
    """Run the interactive stack demonstration menu.

    The demo seeds the stack with a large sample dataset and then lets the user
    test push and pop operations while observing their complexity characteristics.
    """
    stack = Stack()
    print("\n⚙️  Injecting 10,000 baseline items into the Stack...")
    for i in range(10000):
        stack.push(f"item_{i}")
    print("✅ Injection complete!")

    while True:
        print(f"--- STACK OPERATIONS (Elements: {stack.size()} | RAM: {stack.get_memory_bytes()} bytes) ---")
        print("1. Push New Item (Time: O(1) | Space: O(1))")
        print("2. Pop Top Item  (Time: O(1) | Space: O(1))")
        print("0. Back to Main Menu")
        choice = input("Select an operation: ").strip()

        if choice == '1':
            val = input("Enter value to push: ").strip()
            if val:
                run_complexity_quiz("Stack Push", "O(1)", "O(1)", stack, lambda: stack.push(val))
        elif choice == '2':
            if stack.is_empty():
                print("\n⚠️ Stack empty!")
            else:
                popped = run_complexity_quiz("Stack Pop", "O(1)", "O(1)", stack, stack.pop)
                print(f"✅ Popped: {popped}")
        elif choice == '0':
            return


def run_queue_demo():
    """Run the interactive queue demonstration menu.

    The queue starts with many elements so the user can compare an efficient append
    to the more costly dequeue operation that shifts the remaining items.
    """
    queue = Queue()
    print("\n⚙️  Injecting 10,000 baseline items into the Queue...")
    for i in range(10000):
        queue.enqueue(f"item_{i}")
    print("✅ Injection complete!")

    while True:
        print(f"--- QUEUE OPERATIONS (Elements: {queue.size()} | RAM: {queue.get_memory_bytes()} bytes) ---")
        print("1. Enqueue New Item (Time: O(1) | Space: O(1))")
        print("2. Dequeue Front Item (Time: O(n) | Space: O(1))")
        print("0. Back to Main Menu")
        choice = input("Select an operation: ").strip()

        if choice == '1':
            val = input("Enter value to enqueue: ").strip()
            if val:
                run_complexity_quiz("Queue Enqueue", "O(1)", "O(1)", queue, lambda: queue.enqueue(val))
        elif choice == '2':
            if queue.is_empty():
                print("\n⚠️ Queue empty!")
            else:
                dequeued = run_complexity_quiz("Queue Dequeue", "O(n)", "O(1)", queue, queue.dequeue)
                print(f"✅ Dequeued: {dequeued}")
        elif choice == '0':
            return


def run_linked_list_demo():
    """Run an interactive demonstration of the doubly linked list."""
    sll = SinglyLinkedList()
    dll = DoublyLinkedList()
    print("\n⚙️  Injecting 10,000 baseline items into the Singly Linked List...")
    for i in range(10000):
        sll.insert_at_head(f"item_{i}")
    print("\n⚙️  Injecting 10,000 baseline items into the Doubly Linked List...")
    for i in range(10000):
        dll.insert_at_head(f"item_{i}")
    print("✅ Injection complete!")

    while True:
        print(
            f"--- SINGLY LINKED LIST (Elements: {sll.size()} | "
            f"RAM: {sll.get_memory_bytes()} bytes) ---"
        )
        print(
            f"--- DOUBLY LINKED LIST (Elements: {dll.size()} | "
            f"RAM: {dll.get_memory_bytes()} bytes) ---"
        )
        print("1. SinglyLinkedList Insert at Head (Time: O(1) | Space: O(1))")
        print("2. SinglyLinkedList Search for Value (Time: O(n) | Space: O(1))")
        print("3. SinglyLinkedList Delete Value (Time: O(n) | Space: O(1))")
        print("4. DoublyLinkedList Insert at Head (Time: O(1) | Space: O(1))")
        print("5. DoublyLinkedList Insert at Tail (Time: O(1) | Space: O(1))")
        print("6. DoublyLinkedList Search for Value (Time: O(n) | Space: O(1))")
        print("7. DoublyLinkedList Delete Value (Time: O(n) | Space: O(1))")
        print("8. DoublyLinkedList Traverse Forward and Backward")
        print("9. Back to Main Menu")
        choice = input("Select an operation: ").strip()

        if choice == '1':
            val = input("Enter value to insert in Singly Linked List: ").strip()
            if val:
                run_complexity_quiz(
                    "SLL Insert at Head", "O(1)", "O(1)", sll,
                    lambda: sll.insert_at_head(val),
                )
        elif choice == '2':
            val = input("Enter value to search for in Singly Linked List: ").strip()
            if val:
                run_complexity_quiz(
                    "SLL Search for Value", "O(n)", "O(1)", sll,
                    lambda: sll.search(val),
                )
        elif choice == '3':
            target = input("Enter value to delete from Singly Linked List: ").strip()
            if target:
                found = run_complexity_quiz(
                    "SLL Delete", "O(n)", "O(1)", sll,
                    lambda: sll.delete(target),
                )
                print(f"{'✅ Found' if found else '❌ Not found'}: {target}")
        elif choice == '4':
            target = input("Enter value to insert in Doubly Linked List: ").strip()
            if target:
                run_complexity_quiz(
                    "DLL Insert at Head", "O(1)", "O(1)", dll,
                    lambda: dll.insert_at_head(target),
                )
        elif choice == '5':
            target = input("Enter value to insert at tail of Doubly Linked List: ").strip()
            if target:
                run_complexity_quiz(
                    "DLL Insert at Tail", "O(1)", "O(1)", dll,
                    lambda: dll.insert_at_tail(target),
                )
            forward = []
            current = dll.head
            while current:
                forward.append(current.data)
                current = current.next

            backward = []
            current = dll.tail
            while current:
                backward.append(current.data)
                current = current.prev

            def preview(values):
                if len(values) <= 10:
                    return values
                return values[:5] + ["..."] + values[-5:]

            print(f"Forward (head to tail): {preview(forward)}")
            print(f"Backward (tail to head): {preview(backward)}")
        elif choice == '6':
            target = input("Enter value to search for in Doubly Linked List: ").strip()
            if target:
                run_complexity_quiz(
                    "DLL Search for Value", "O(n)", "O(1)", dll,
                    lambda: dll.search(target),
                )
        elif choice == '7':
            target = input("Enter value to delete from Doubly Linked List: ").strip()
            if target:
                found = run_complexity_quiz(
                    "DLL Delete", "O(n)", "O(1)", dll,
                    lambda: dll.delete(target),
                )
                print(f"{'✅ Found' if found else '❌ Not found'}: {target}")
        elif choice == '8':
            forward = []
            current = dll.head
            while current:
                forward.append(current.data)
                current = current.next

            backward = []
            current = dll.tail
            while current:
                backward.append(current.data)
                current = current.prev

            def preview(values):
                if len(values) <= 10:
                    return values
                return values[:5] + ["..."] + values[-5:]

            print(f"Forward (head to tail): {preview(forward)}")
            print(f"Backward (tail to head): {preview(backward)}")
        elif choice == '9':
            break


# ==============================================================================
# LINEAR DATA STRUCTURES REAL-WORLD ALGORITHMS & APPLICATIONS
# ==============================================================================

def is_balanced(expression: str) -> bool:
    """
    Validates if compiler brackets/parentheses inside a string are properly matching.
    
    Uses a Stack to retain unclosed opening brackets in LIFO order.
    
    Args:
        expression: Target code or string containing brackets to check.
        
    Example:
        >>> is_balanced("{[()]}") -> True
        >>> is_balanced("{[(])}") -> False
    """
    stack = Stack()
    mapping: Dict[str, str] = {")": "(", "}": "{", "]": "["}
    
    for char in expression:
        if char in mapping.values():
            stack.push(char)
        elif char in mapping.keys():
            if stack.is_empty() or stack.pop() != mapping[char]:
                return False
    return stack.is_empty()


def process_print_jobs(job_list: List[str]) -> List[str]:
    """
    Simulates a network spooler queue processing operational print streams.
    
    Uses a standard Queue to process transactions based on arrival timelines (FIFO).
    
    Args:
        job_list: Linear list of incoming filenames to spool.
    """
    queue = Queue()
    log: List[str] = []
    
    for job in job_list:
        queue.enqueue(job)
        
    while not queue.is_empty():
        current_job = queue.dequeue()
        log.append(f"Completed: {current_job}")
    return log


def is_palindrome(text: str) -> bool:
    """
    Determines if a clean alphanumeric sequence reads the same backwards and forwards.

    Uses a Deque to evaluate letters symmetrically from both ends inward.

    Args:
        text: Target sentence or word to validate.
    """
    clean_text = "".join(char.lower() for char in text if char.isalnum())
    char_deque = Deque()

    for char in clean_text:
        char_deque.add_rear(char)

    while char_deque.size() > 1:
        if char_deque.remove_front() != char_deque.remove_rear():
            return False

    return True



def run_sample_linear_data_structures_test_suite() -> None:
    """Demonstrate each data structure and its applications with sample data."""
    print("\n=== Linear Data Structures Sample Tests ===")

    stack = Stack()
    for item in ["first", "second", "third"]:
        stack.push(item)
    print(
        f"Stack: {stack.to_list()}; size={stack.size()}; "
        f"peek={stack.peek()}; pop={stack.pop()}"
    )

    queue = Queue()
    for item in ["Alice", "Bob", "Carol"]:
        queue.enqueue(item)
    print(
        f"Queue: {queue.to_list()}; size={queue.size()}; "
        f"front={queue.front()}; dequeue={queue.dequeue()}"
    )

    deque = Deque()
    deque.add_front("middle")
    deque.add_front("front")
    deque.add_rear("rear")
    print(f"Deque: {deque.to_list()}; remove_front={deque.remove_front()}; "
          f"remove_rear={deque.remove_rear()}")

    singly_linked_list = SinglyLinkedList()
    for item in ["A", "B", "C"]:
        singly_linked_list.insert_at_head(item)
    singly_linked_list.delete("B")
    singly_linked_list.insert_at_head("D")
    print(
        f"Singly LinkedList: {singly_linked_list.display()}; "
        f"search('A')={singly_linked_list.search('A')}; "
        f"empty={singly_linked_list.is_empty()}"
    )

    print(f"Stack applications - Balanced brackets for '{{[()]}}': {is_balanced('{[()]}')}")
    print(f"Queue applications - Process Print jobs for ['report.pdf', 'photo.jpg'] (FIFO`): {process_print_jobs(['report.pdf', 'photo.jpg'])}")
    print(f"Deque applications - Palindrome for 'Never odd or even': {is_palindrome('Never odd or even')}")




class PlaylistManager:
    """
    Manages an active audio playlist pipeline using a DoublyLinkedList core.
    Demonstrates ideal usage of DoublyLinkedLists, where nodes (tracks) are frequently
    inserted or skipped dynamically without rebuilding structural indices.
    """

    def __init__(self) -> None:
        """Initializes a new playlist interface containing an empty DoublyLinkedList."""
        self.playlist = DoublyLinkedList()

    def add_song(self, title: str) -> None:
        """Adds a track name straight to the playlist structure."""
        self.playlist.insert_at_head(title)

    def remove_song(self, title: str) -> bool:
        """Deletes a designated song track. Returns True if removed successfully."""
        return self.playlist.delete(title)

    def has_song(self, title: str) -> bool:
        """Return True if the playlist contains the specified song."""
        return self.playlist.search(title)

    def get_playlist_size(self) -> int:
        """Return the number of tracks currently in the playlist."""
        return self.playlist.size()

    def clear_playlist(self) -> None:
        """Remove all tracks from the playlist."""
        self.playlist.clear()

    def has_next_song(self, current_title: str) -> bool:
        """Return True if there is a subsequent track after the current one."""
        current_node = self.playlist.head
        while current_node:
            if current_node.data == current_title:
                return current_node.next is not None
            current_node = current_node.next
        return False

    def has_previous_song(self, current_title: str) -> bool:
        """Return True if there is a preceding track before the current one."""
        current_node = self.playlist.head
        while current_node:
            if current_node.data == current_title:
                return current_node.prev is not None
            current_node = current_node.next
        return False

    def show_playlist(self) -> str:
        """Outputs an explicitly mapped display sequence representing the tracklist."""
        return self.playlist.display()



def run_linear_data_structure_use_case_examples() -> None:
    """Explain each data structure with a familiar everyday example."""
    print("\n=== Problem-Solving Examples ===")

    print("\n1. Stack - a pile of plates")
    print("   Imagine stacking plates in a cupboard. The last plate placed on the pile")
    print("   is the first plate you take off. This is useful when checking nested")
    print("   items, such as making sure every opening bracket has a matching bracket.")
    expression = "{[()]}"
    bracket_stack = Stack()
    for bracket in expression:
        if bracket in "{[(":
            bracket_stack.push(bracket)
            print(f"   Put '{bracket}' on the stack: {bracket_stack.to_list()}")
        else:
            opening_bracket = bracket_stack.pop()
            print(
                f"   Found '{bracket}', so remove '{opening_bracket}' from the stack: "
                f"{bracket_stack.to_list()}"
            )
    print(f"   Example: {expression} has matching brackets: {is_balanced(expression)}")

    print("\n2. Queue - a line at a store")
    print("   Imagine customers waiting at a checkout counter. The person who arrives")
    print("   first is helped first. This is useful for handling print jobs, customers,")
    print("   or other requests in the order they arrive.")
    jobs = ["family_photo.jpg", "grocery_list.txt", "birthday_card.pdf"]
    print("   Add each print job to the back of the line:")
    job_queue = Queue()
    for job in jobs:
        job_queue.enqueue(job)
        print(f"   Add {job}: {job_queue.to_list()}")
    print("   Remove and complete jobs from the front of the line:")
    while not job_queue.is_empty():
        print(f"   Complete {job_queue.dequeue()}; remaining jobs: {job_queue.to_list()}")

    print("\n3. Deque - a line with two entrances")
    print("   Imagine a line where people can join at either end and leave from either")
    print("   end. This is useful when comparing the first and last items, such as")
    print("   checking whether a word or phrase reads the same in both directions.")
    text = "Never odd or even"
    letter_deque = Deque()
    for character in "neveroddoreven":
        letter_deque.add_rear(character)
    print(f"   Add letters to the rear: {letter_deque.to_list()}")
    print("   Compare and remove one letter from each end:")
    while letter_deque.size() > 1:
        left = letter_deque.remove_front()
        right = letter_deque.remove_rear()
        print(f"   Compare '{left}' and '{right}' -> remaining: {letter_deque.to_list()}")
    print(f"   Example: '{text}' reads the same both ways: {is_palindrome(text)}")

    print("\n4. Linked list - a chain of connected songs")
    print("   Imagine songs connected in playlist order. A new song can be added, or")
    print("   a song can be removed, without rebuilding the entire playlist.")
    print("   This is useful when people frequently update the songs they want to hear.")
    playlist = PlaylistManager()
    for song in ["Morning Song", "Travel Song", "Evening Song"]:
        playlist.add_song(song)
        print(f"   Add '{song}': {playlist.show_playlist()}")
    print(f"   Playlist before removal: {playlist.show_playlist()}")
    print(f"   Search for 'Travel Song': {playlist.has_song('Travel Song')}")
    playlist.remove_song("Travel Song")
    print(f"   Remove 'Travel Song': {playlist.show_playlist()}")


def _measure_operation(operation: Callable[[], None]) -> float:
    """Return the elapsed time, in milliseconds, for one operation batch."""
    start = perf_counter()
    operation()
    return (perf_counter() - start) * 1000


def run_performance_comparison(item_count: int = 5_000) -> None:
    """Display theoretical and measured efficiency for each data structure.

    Timings are illustrative and depend on the computer running the program.
    The Big-O descriptions explain how the operations grow as the data size
    increases and are more meaningful than any single timing result.
    """
    print("\n--- Performance Comparison ---")
    print("Array: append and random access are O(1) amortized; best for indexed storage and retrieval.")
    print("Stack: push/pop/peek are O(1) amortized; best for nested operations such as matching brackets.")
    print("Queue: enqueue is O(1) amortized, dequeue is O(n) in a Python list; best for FIFO scheduling.")
    print("Deque: front/rear insertions/removals vary; best for symmetric access from both ends, such as palindrome checks.")
    print("LinkedList: insert is O(1), while search/delete are O(n); best for dynamic insertion and removal without resizing arrays.")

    print(f"\nMeasured time for batches of {item_count:,} operations (milliseconds):")
    print(f"{'Implementation':<18} {'Operation batch':<28} {'Time':>10}")
    print("-" * 60)

    def array_batch() -> None:
        array = Array()
        for value in range(item_count):
            array.append(value)
        for index in range(item_count):
            _ = array.get(index)

    def stack_batch() -> None:
        stack = Stack()
        for value in range(item_count):
            stack.push(value)
        while not stack.is_empty():
            stack.pop()

    def queue_batch() -> None:
        queue = Queue()
        for value in range(item_count):
            queue.enqueue(value)
        while not queue.is_empty():
            queue.dequeue()

    def deque_front_batch() -> None:
        deque = Deque()
        for value in range(item_count):
            deque.add_rear(value)
        while not deque.is_empty():
            deque.remove_front()

    def singly_linked_list_batch() -> None:
        singly_linked_list = SinglyLinkedList()
        for value in range(item_count):
            singly_linked_list.insert_at_head(value)
        singly_linked_list.search(0)  # Searches to the tail: a worst-case O(n) search.

    def doubly_linked_list_batch() -> None:
        doubly_linked_list = DoublyLinkedList()
        for value in range(item_count):
            doubly_linked_list.insert_at_head(value)
        doubly_linked_list.search(0)  # Searches to the tail: a worst-case O(n) search.

    measurements = [
        ("Array", "append n + get n", array_batch),
        ("Stack", "push n + pop n", stack_batch),
        ("Queue", "enqueue n + dequeue n", queue_batch),
        ("Deque", "add rear n + remove front n", deque_front_batch),
        ("SinglyLinkedList", "insert n + search tail", singly_linked_list_batch),
        ("DoublyLinkedList", "insert at head n + search tail", doubly_linked_list_batch),
    ]
    for name, operation_batch, operation in measurements:
        elapsed = _measure_operation(operation)
        print(f"{name:<18} {operation_batch:<28} {elapsed:>10.3f}")


def _read_value(prompt: str) -> str:
    """Read a non-empty value for the interactive demonstrations."""
    value = input(prompt).strip()
    return value if value else "(empty)"


def run_interactive_linear_data_structures_test_suite() -> None:
    """Run an interactive demonstration of every supported data structure."""
    while True:
        print("\n=== Linear Data Structures Test Suite ===")
        print("1. Test Stack")
        print("2. Test Queue")
        print("3. Test Deque")
        print("4. Test LinkedList")
        print("5. Run application examples")
        print("6. Show performance comparison")
        print("7. Run Interactive Functionality Tests (Balanced Brackets, Palindrome, Print Jobs)")
        print("8. Run Complexity Quiz")
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            stack = Stack()
            for item in input("Enter stack items (space-separated): ").split():
                stack.push(item)
            print(f"After push: {stack.to_list()}")
            if not stack.is_empty():
                print(f"Peek: {stack.peek()}")
                print(f"Pop: {stack.pop()}")
            print(f"Remaining stack: {stack.to_list()}; empty={stack.is_empty()}")
        elif choice == "2":
            queue = Queue()
            for item in input("Enter queue items (space-separated): ").split():
                queue.enqueue(item)
            if not queue.is_empty():
                print(f"Front: {queue.front()}")
                print(f"Dequeue: {queue.dequeue()}")
            print(f"Remaining queue: {queue.to_list()}; empty={queue.is_empty()}")
        elif choice == "3":
            deque_structure = Deque()
            front_items = [
                item.strip()
                for item in input("Items to add at front (comma-separated): ").split(",")
                if item.strip()
            ]
            for item in reversed(front_items):
                deque_structure.add_front(item)
            rear_items = [
                item.strip()
                for item in input("Items to add at rear (comma-separated): ").split(",")
                if item.strip()
            ]
            for item in rear_items:
                deque_structure.add_rear(item)
            print(f"Deque after additions: {deque_structure.to_list()}")
            print(f"Remove front: {deque_structure.remove_front()}")
            print(f"Remove rear: {deque_structure.remove_rear()}")
            print(f"Remaining deque: {deque_structure.to_list()}; empty={deque_structure.is_empty()}")
        elif choice == "4":
            singly_linked_list = SinglyLinkedList()
            items = input("Enter linked-list items (comma- or space-separated): ")
            for item in items.replace(",", " ").split():
                singly_linked_list.insert_at_head(item)
            print(f"List (newest first): {singly_linked_list.display()}")
            key = _read_value("Value to search and delete: ").strip("\"'")
            print(f"Search result: {singly_linked_list.search(key)}")
            print(f"Delete result: {singly_linked_list.delete(key)}")
            print(f"List after delete: {singly_linked_list.display()}")
        elif choice == "5":
            doubly_linked_list = DoublyLinkedList()
            items = input("Enter doubly-linked-list items (comma- or space-separated): ")
            for item in items.replace(",", " ").split():
                doubly_linked_list.insert_at_head(item)
            print(f"List (newest first): {doubly_linked_list.display()}")
            key = _read_value("Value to search and delete: ").strip("\"'")
            print(f"Search result: {doubly_linked_list.search(key)}")
            print(f"Delete result: {doubly_linked_list.delete(key)}")
            print(f"List after delete: {doubly_linked_list.display()}")
        elif choice == "6":
            doubly_linked_list = DoublyLinkedList()
            items = input("Enter doubly-linked-list items (comma- or space-separated): ")
            for item in items.replace(",", " ").split():
                doubly_linked_list.insert_at_tail(item)
            print(f"List (newest first): {doubly_linked_list.display()}")
            key = _read_value("Value to search and delete: ").strip("\"'")
            print(f"Search result: {doubly_linked_list.search(key)}")
            print(f"Delete result: {doubly_linked_list.delete(key)}")
            print(f"List after delete: {doubly_linked_list.display()}")
        elif choice == "7":
            print(f"Balanced brackets: {is_balanced(_read_value('Enter a bracket Expression: '))}")
            print(f"Palindrome: {is_palindrome(_read_value('Enter text to test for palindrome: '))}")
            jobs = input("Enter tasks to be done as a Queue - customer line (FIFO - first job received gets to be completed first): ").split()
            print("\n".join(process_print_jobs(jobs)) or "No jobs supplied.")
        elif choice == "8":
            item_count_str = input("Enter number of items for performance comparison (default 10000): ").strip()
            item_count = int(item_count_str) if item_count_str.isdigit() else 10000
            run_performance_comparison(item_count)
        if choice == "0":
                print("Test suite complete. Exiting to main menu.")
                return
        else:
            print("Please choose 0-8.")

def linear_data_structure_implementation():
    """Display the main menu and manage program flow.

    The loop keeps the learning tool active until the user chooses to exit.
    """
    while True:
        print("\n========================================================")
        print("    LINEAR DATA STRUCTURE SCALED TIME & SPACE BENCHMARK TOOL   ")
        print("========================================================")
        print("1. Explore Arrays (Random Access - Shallow Memory Diagnostics)")
        print("2. Explore Stack (LIFO - Deep Memory Diagnostics)")
        print("3. Explore Queue (FIFO - Deep Memory Diagnostics)")
        print("4. Explore Linked List (Sequential - Safe Traversal)")
        print("5. Run Sample Data Test Suite")
        print("6. Run Problem-Solving Use Case Examples")
        print("7. Run Performance Comparison")
        print("8. Run Interactive Linear Data Structure test suite Demo")
        print("9. Exit Application")
        main_choice = input("Choose a data structure to analyze: ").strip()

        if main_choice == '1':
            run_array_demo()
        elif main_choice == '2':
            run_stack_demo()
        elif main_choice == '3':
            run_queue_demo()
        elif main_choice == '4':
            run_linked_list_demo()
        elif main_choice == '5':
            run_sample_linear_data_structures_test_suite()
        elif main_choice == '6':
            run_linear_data_structure_use_case_examples()
        elif main_choice == '7':
            run_performance_comparison()
        elif main_choice == '8':
            run_interactive_linear_data_structures_test_suite()
        elif main_choice == '9':
            print("\nExiting metrics engine. Happy engineering!")
            sys.exit()



# ================================================================
# END LINEAR DATA STRUCTURE IMPLEMENTATION
# ================================================================








# ================================================================
# START - NON-LINEAR DATA STRUCTURE USER INTERFACE
# ================================================================

# =====================================================================
# CORE BINARY SEARCH TREE IMPLEMENTATION
# =====================================================================

class BSTNode:
    """Represents a structural node inside the Binary Search Tree."""
    def __init__(self, key: Any, value: Any):
        self.key: Any = key
        self.value: Any = value
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None


class BinarySearchTree:
    """
    A foundational Binary Search Tree class implementing core mutations,
    lookups, traversal iterators, min/max metrics, and balance evaluations.
    """
    def __init__(self):
        self.root: Optional[BSTNode] = None

    def insert(self, key: Any, value: Any) -> None:
        """Inserts a key-value node into the BST. Updates value if key exists."""
        self.root = self._insert_recursive(self.root, key, value)
        self._rebalance_if_needed()

    def _insert_recursive(self, node: Optional[BSTNode], key: Any, value: Any) -> BSTNode:
        if not node:
            return BSTNode(key, value)
        
        if key == node.key:
            node.value = value
        elif key < node.key:
            node.left = self._insert_recursive(node.left, key, value)
        else:
            node.right = self._insert_recursive(node.right, key, value)
        return node

    def search(self, key: Any) -> Optional[BSTNode]:
        """Searches for a key within the tree. Returns the node if found, else None."""
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node: Optional[BSTNode], key: Any) -> Optional[BSTNode]:
        if not node or node.key == key:
            return node
        if key < node.key:
            return self._search_recursive(node.left, key)
        return self._search_recursive(node.right, key)

    def delete(self, key: Any) -> None:
        """Removes a key from the tree, maintaining proper BST properties."""
        self.root = self._delete_recursive(self.root, key)
        self._rebalance_if_needed()

    def _delete_recursive(self, node: Optional[BSTNode], key: Any) -> Optional[BSTNode]:
        if not node:
            return None

        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # Node with only one child or no children
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            # Node with two children: Fetch inorder successor (smallest in right subtree)
            successor = self._find_min_node(node.right)
            node.key = successor.key
            node.value = successor.value
            node.right = self._delete_recursive(node.right, successor.key)

        return node

    def _rebalance_if_needed(self) -> None:
        if not self.is_balanced():
            nodes = [(node.key, node.value) for node in self.traverse_in_order()]
            self.root = self._build_balanced_tree(nodes, 0, len(nodes) - 1)

    def _build_balanced_tree(
        self, nodes: List[Tuple[Any, Any]], first: int, last: int
    ) -> Optional[BSTNode]:
        if first > last:
            return None

        midpoint = (first + last) // 2
        key, value = nodes[midpoint]
        node = BSTNode(key, value)
        node.left = self._build_balanced_tree(nodes, first, midpoint - 1)
        node.right = self._build_balanced_tree(nodes, midpoint + 1, last)
        return node

    # --- TREE MANIPULATION ALGORITHMS (MIN/MAX) ---
    def find_min(self) -> Optional[Tuple[Any, Any]]:
        """Returns the minimum (key, value) pair in the tree."""
        if not self.root:
            return None
        min_node = self._find_min_node(self.root)
        return (min_node.key, min_node.value)

    def _find_min_node(self, node: BSTNode) -> BSTNode:
        current = node
        while current.left:
            current = current.left
        return current

    def find_max(self) -> Optional[Tuple[Any, Any]]:
        """Returns the maximum (key, value) pair in the tree."""
        if not self.root:
            return None
        current = self.root
        while current.right:
            current = current.right
        return (current.key, current.value)


# =====================================================================
# TREE TRAVERSAL ALGORITHMS
# =====================================================================
    def traverse_in_order(self) -> Generator[BSTNode, None, None]:
        """Yields nodes via an in-order traversal path (Left, Root, Right)."""
        yield from self._in_order_recursive(self.root)

    def _in_order_recursive(self, node: Optional[BSTNode]) -> Generator[BSTNode, None, None]:
        if node:
            yield from self._in_order_recursive(node.left)
            yield node
            yield from self._in_order_recursive(node.right)

    def traverse_pre_order(self) -> Generator[BSTNode, None, None]:
        """Yields nodes via a pre-order traversal path (Root, Left, Right)."""
        yield from self._pre_order_recursive(self.root)

    def _pre_order_recursive(self, node: Optional[BSTNode]) -> Generator[BSTNode, None, None]:
        if node:
            yield node
            yield from self._pre_order_recursive(node.left)
            yield from self._pre_order_recursive(node.right)

    def traverse_post_order(self) -> Generator[BSTNode, None, None]:
        """Yields nodes via a post-order traversal path (Left, Right, Root)."""
        yield from self._post_order_recursive(self.root)

    def _post_order_recursive(self, node: Optional[BSTNode]) -> Generator[BSTNode, None, None]:
        if node:
            yield from self._post_order_recursive(node.left)
            yield from self._post_order_recursive(node.right)
            yield node

# =====================================================================
# TREE BALANCING DETECTION SYSTEM
# =====================================================================
    def get_height(self) -> int:
        """Calculates the total height of the tree structure."""
        return self._calculate_height(self.root)

    def _calculate_height(self, node: Optional[BSTNode]) -> int:
        if not node:
            return 0
        return 1 + max(self._calculate_height(node.left), self._calculate_height(node.right))

    def is_balanced(self) -> bool:
        """
        Evaluates structural layout balance factor requirements.
        A tree is balanced if heights of left and right subtrees differ by at most 1.
        """
        balanced, _ = self._check_balance_recursive(self.root)
        return balanced

    def validate_bst_property(self) -> bool:
        """Returns True when every node is ordered correctly in the BST."""
        def validate(node: Optional[BSTNode], lower: Any = None, upper: Any = None) -> bool:
            if not node:
                return True
            if lower is not None and node.key <= lower:
                return False
            if upper is not None and node.key >= upper:
                return False
            return (
                validate(node.left, lower, node.key)
                and validate(node.right, node.key, upper)
            )

        return validate(self.root)

    def _check_balance_recursive(self, node: Optional[BSTNode]) -> Tuple[bool, int]:
        if not node:
            return True, 0

        left_balanced, left_height = self._check_balance_recursive(node.left)
        right_balanced, right_height = self._check_balance_recursive(node.right)

        # A node is unbalanced if subtrees differ by more than 1, or subtrees themselves are unbalanced
        current_balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
        current_height = 1 + max(left_height, right_height)

        return current_balanced, current_height

    def visualize(self) -> str:
        """Returns a compact sideways ASCII representation of the tree."""
        if not self.root:
            return "(empty tree)"

        lines: List[str] = []

        def render(node: Optional[BSTNode], prefix: str, branch: str) -> None:
            if not node:
                return
            if node.right:
                render(node.right, prefix + "    ", "R---")
            lines.append(f"{prefix}{branch}{node.key}")
            if node.left:
                render(node.left, prefix + "    ", "L---")

        render(self.root, "", "ROOT ")
        return "\n".join(lines)


# =====================================================================
#  MAP CLASS IMPLEMENTATION
# =====================================================================

class TreeMap:
    """A map-like key-value data structure implemented using an underlying BST."""
    def __init__(self):
        self._bst = BinarySearchTree()
        self._size = 0

    def __setitem__(self, key: Any, value: Any) -> None:
        """Allows item assignment syntax: my_map[key] = value"""
        # Track structural mutations to verify size changes accurately
        existing = self._bst.search(key)
        if not existing:
            self._size += 1
        self._bst.insert(key, value)

    def __getitem__(self, key: Any) -> Any:
        """Allows collection lookup syntax: value = my_map[key]"""
        node = self._bst.search(key)
        if not node:
            raise KeyError(f"Key '{key}' not found in TreeMap.")
        return node.value

    def __delitem__(self, key: Any) -> None:
        """Allows deletion syntax: del my_map[key]"""
        node = self._bst.search(key)
        if not node:
            raise KeyError(f"Key '{key}' not found in TreeMap.")
        self._bst.delete(key)
        self._size -= 1

    def __contains__(self, key: Any) -> bool:
        """Supports conditional containment: 'if key in my_map:'"""
        return self._bst.search(key) is not None

    def __len__(self) -> int:
        """Returns tracking count size elements."""
        return self._size

    def get(self, key: Any, default: Any = None) -> Any:
        """Safely fetches items from map matching standard fallback protocols."""
        node = self._bst.search(key)
        return node.value if node else default

    def keys(self) -> List[Any]:
        """Returns sorted keys extracted utilizing in-order sorting capabilities."""
        return [node.key for node in self._bst.traverse_in_order()]

    def values(self) -> List[Any]:
        """Returns values matching key assignments sorted through in-order traversal."""
        return [node.value for node in self._bst.traverse_in_order()]

    def pre_order(self) -> List[Any]:
        """Returns keys in pre-order traversal order."""
        return [node.key for node in self._bst.traverse_pre_order()]

    def post_order(self) -> List[Any]:
        """Returns keys in post-order traversal order."""
        return [node.key for node in self._bst.traverse_post_order()]

    def find_min(self) -> Optional[Tuple[Any, Any]]:
        """Returns the minimum key-value pair in the map."""
        return self._bst.find_min()

    def find_max(self) -> Optional[Tuple[Any, Any]]:
        """Returns the maximum key-value pair in the map."""
        return self._bst.find_max()

    def visualize(self) -> str:
        """Returns an ASCII representation of the underlying tree."""
        return self._bst.visualize()

    def validate_bst_property(self) -> bool:
        """Returns True when the underlying tree is correctly ordered."""
        return self._bst.validate_bst_property()

    def verify_balance_status(self) -> str:
        """Exposes the underlying tree metrics tracking imbalances."""
        balanced = self._bst.is_balanced()
        height = self._bst.get_height()
        return f"Balanced: {balanced} (Tree Height: {height})"


# =====================================================================
# INTERACTIVE DEMONSTRATION INTERFACE
# =====================================================================

def populate_balanced_structure(map_target: TreeMap, items: List[Tuple[Any, Any]]) -> None:
    """Inserts sorted items in midpoint order to produce a short tree."""
    if not items:
        return
    midpoint = len(items) // 2
    node_key, node_value = items[midpoint]
    map_target[node_key] = node_value
    populate_balanced_structure(map_target, items[:midpoint])
    populate_balanced_structure(map_target, items[midpoint + 1:])


def make_sample_map(keys: List[int]) -> TreeMap:
    """Build a balanced sample map from a list of integer keys."""
    sample_map = TreeMap()
    populate_balanced_structure(sample_map, [(key, f"value-{key}") for key in sorted(keys)])
    return sample_map


def make_payload(index: int) -> Any:
    """Create a varied payload value for testing mixed-type map data."""
    value_type = index % 4
    if value_type == 0:
        return f"item_{index}"
    if value_type == 1:
        return index * 1.5
    if value_type == 2:
        return [index, index + 1]
    return {"data_index": index, "even_flag": index % 2 == 0}


class TreeDemoInterface:
    """Terminal interface for exploring the BST and its performance."""

    def run(self) -> None:
        print("\n=== Binary Search Tree Map Explorer ===")
        print("Choose a demonstration to run.")
        while True:
            print("\n1. Traversal algorithms")
            print("2. Construction, search, and manipulation")
            print("3. Tree map vs. list map performance")
            print("4. Visualize insertions and deletions")
            print("5. Run all demonstrations")
            print("6. Evaluate benefits and use cases")
            print("7. Test 50 mixed-type data items")
            print("0. Exit")
            try:
                choice = input("\nSelect an option: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                return

            actions = {
                "1": self.show_traversals,
                "2": self.show_manipulation,
                "3": self.show_performance,
                "4": self.show_visualization,
                "5": self.show_all,
                "6": self.show_evaluation,
                "7": self.show_mixed_type_test,
            }
            if choice == "0":
                print("Session ended.")
                return
            action = actions.get(choice)
            if action:
                action()
            else:
                print("Please choose a number from 0 to 7.")

    def show_traversals(self) -> None:
        """Display preorder, inorder, and postorder traversal results."""
        sample_map = make_sample_map([50, 30, 70, 20, 40, 60, 80])
        print("\n--- Traversal Algorithms ---")
        print("Tree keys:         [50, 30, 70, 20, 40, 60, 80]")
        print(f"Pre-order (Root, Left, Right): {sample_map.pre_order()}")
        print(f"In-order (Left, Root, Right):  {sample_map.keys()}")
        print(f"Post-order (Left, Right, Root): {sample_map.post_order()}")

    def show_manipulation(self) -> None:
        sample_map = TreeMap()
        for key in [50, 30, 70, 20, 40, 60, 80]:
            sample_map[key] = f"value-{key}"
        print("\n--- Construction, Search, and Manipulation ---")
        print(f"Constructed keys: {sample_map.keys()}")
        print(f"Search for 60: {sample_map.get(60)}")
        print(f"Search for 99: {sample_map.get(99, 'not found')}")
        sample_map[65] = "value-65"
        print(f"After inserting 65: {sample_map.keys()}")
        del sample_map[30]
        print(f"After deleting 30: {sample_map.keys()}")
        print(f"Minimum: {sample_map.find_min()} | Maximum: {sample_map.find_max()}")
        print(f"Balance status: {sample_map.verify_balance_status()}")
        print(f"BST property valid: {sample_map.validate_bst_property()}")

    def show_performance(self) -> None:
        item_count = 2000
        repetitions = 500
        items = [(index, f"value-{index}") for index in range(item_count)]
        tree_map = TreeMap()
        populate_balanced_structure(tree_map, items)
        list_map = items
        lookup_keys = [0, item_count // 2, item_count - 1, item_count + 1] * repetitions

        tree_start = time.perf_counter()
        for lookup_key in lookup_keys:
            tree_map.get(lookup_key)
        tree_elapsed = time.perf_counter() - tree_start

        list_start = time.perf_counter()
        for lookup_key in lookup_keys:
            next((value for key, value in list_map if key == lookup_key), None)
        list_elapsed = time.perf_counter() - list_start

        print("\n--- Search Performance ---")
        print(f"Dataset: {item_count} integer keys, {len(lookup_keys)} searches")
        print(f"TreeMap search time: {tree_elapsed:.6f} seconds")
        print(f"ListMap search time: {list_elapsed:.6f} seconds")
        if tree_elapsed:
            print(f"List/Tree time ratio: {list_elapsed / tree_elapsed:.2f}x")
        print("TreeMap uses height-based search; ListMap scans entries sequentially.")
        print("Best use: repeated ordered lookups with changing data and min/max queries.")
        print("Prefer a list for tiny datasets or workloads dominated by simple iteration.")

    def show_visualization(self) -> None:
        """Demonstrate how the BST changes as keys are inserted and removed."""
        sample_map = TreeMap()
        print("\n--- Tree Structure Visualization ---")
        for key in [50, 30, 70, 20, 40, 60, 80]:
            sample_map[key] = f"value-{key}"
            print(f"\nAfter inserting {key}:")
            print(sample_map.visualize())
        del sample_map[30]
        print("\nAfter deleting 30:")
        print(sample_map.visualize())

    def show_all(self) -> None:
        self.show_traversals()
        self.show_manipulation()
        self.show_performance()
        self.show_visualization()
        self.show_mixed_type_test()

    def show_evaluation(self) -> None:
        """Summarize the BST map trade-offs and practical use cases."""
        print("\n--- Evaluation and Appropriate Use Cases ---")
        print("Advantages: approximately O(log n) search in a balanced tree, ordered traversal, and min/max lookup.")
        print("Tradeoff: rebalancing may cost O(n), and a list can be simpler for tiny or iteration-only workloads.")
        print("Use a tree-based map for dynamic ordered data, range-oriented processing, and repeated key searches.")
        print("Use a list-based map for small collections or when preserving simple sequential data is the priority.")
        print()
        print("Example: managing a library catalog by call number")
        catalog = TreeMap()
        books = [
            (501, "Algorithms in Practice"),
            (120, "Introduction to Databases"),
            (330, "Operating Systems Concepts"),
            (210, "Data Structures and Problem Solving"),
        ]
        for call_number, title in books:
            catalog[call_number] = title

        print(f"Inserted call numbers: {[call_number for call_number, _ in books]}")
        print(f"Ordered catalog: {catalog.keys()}")
        print(f"Book at call number 330: {catalog.get(330)}")
        print(f"First catalog entry: {catalog.find_min()}")
        print(f"Last catalog entry: {catalog.find_max()}")

        catalog[210] = "Data Structures and Algorithms"
        del catalog[501]
        print(f"After updating 210 and removing 501: {catalog.keys()}")
        print("This demonstrates why a tree helps: the catalog stays ordered while books are searched, updated, and removed.")
        print("For a small catalog or simple sequential data, a list may be easier to maintain.")

    def show_mixed_type_test(self) -> None:
        print("\n--- 50-Item Mixed-Type Data Test ---")
        mixed_values = [
            "text",
            42,
            3.14,
            [3, 4],
            {"kind": "mapping"},
            True,
            None,
        ]
        items = TreeMap()
        for key in range(50):
            items[key] = mixed_values[key % len(mixed_values)]

        value_types = sorted({type(items.get(key)).__name__ for key in range(50)})
        initial_checks = {
            "50 items inserted": len(items) == 50,
            "keys remain ordered": items.keys() == list(range(50)),
            "all values are retrievable": all(
            items.get(key) == mixed_values[key % len(mixed_values)]
            for key in range(50)
            ),
            "BST property is valid": items.validate_bst_property(),
            "tree is balanced": items.verify_balance_status().startswith("Balanced: True"),
        }
        print("Inserted 50 items with value types:", ", ".join(value_types))
        print(f"First five items: {[(key, items.get(key)) for key in range(5)]}")
        for check_name, passed in initial_checks.items():
            print(f"{check_name}: {'PASS' if passed else 'FAIL'}")

        items[25] = {"updated": True}
        del items[10]
        mutation_checks = {
            "key 25 updated": items.get(25) == {"updated": True},
            "key 10 deleted": 10 not in items,
            "item count is 49": len(items) == 49,
            "BST property remains valid": items.validate_bst_property(),
            "tree remains balanced": items.verify_balance_status().startswith("Balanced: True"),
        }
        print(f"After updating key 25: {items.get(25)}")
        print(f"After deleting key 10, item count: {len(items)}")
        for check_name, passed in mutation_checks.items():
            print(f"{check_name}: {'PASS' if passed else 'FAIL'}")
        all_checks_passed = all(initial_checks.values()) and all(mutation_checks.values())
        print(f"Mixed-type test result: {'PASS' if all_checks_passed else 'FAIL'}")




# =====================================================================
# GRAPH IMPLEMENTATION
# =====================================================================
class Graph:
    def __init__(self, is_directed: bool = False):
        """Initializes a graph that maintains both adjacency-list and adjacency-matrix representations."""
        self.is_directed: bool = is_directed
        self.adj_list: Dict[str, List[Tuple[str, float]]] = {}
        self.nodes: Set[str] = set()
        self.node_order: List[str] = []
        self.index_map: Dict[str, int] = {}
        self.matrix: List[List[Optional[float]]] = []

    def _rebuild_matrix(self) -> None:
        """Synchronizes the adjacency matrix with the current adjacency-list state."""
        self.node_order = sorted(self.nodes)
        self.index_map = {node: index for index, node in enumerate(self.node_order)}
        size = len(self.node_order)
        self.matrix = [[None for _ in range(size)] for _ in range(size)]

        for node in self.node_order:
            node_index = self.index_map[node]
            for neighbor, weight in self.adj_list.get(node, []):
                if neighbor in self.index_map:
                    neighbor_index = self.index_map[neighbor]
                    self.matrix[node_index][neighbor_index] = float(weight)

    def add_node(self, node: str) -> bool:
        """Adds a new vertex to the graph. Returns True if successfully added."""
        if not node.strip():
            return False
        if node in self.nodes:
            return False

        self.nodes.add(node)
        self.adj_list[node] = []
        self._rebuild_matrix()
        return True

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        """Adds an edge between node u and node v with an optional weight."""
        self.add_node(u)
        self.add_node(v)

        self.adj_list[u] = [(neighbor, edge_weight) for neighbor, edge_weight in self.adj_list.get(u, []) if neighbor != v]
        self.adj_list[u].append((v, float(weight)))

        if not self.is_directed:
            self.adj_list[v] = [(neighbor, edge_weight) for neighbor, edge_weight in self.adj_list.get(v, []) if neighbor != u]
            self.adj_list[v].append((u, float(weight)))

        self._rebuild_matrix()

    def remove_edge(self, u: str, v: str) -> bool:
        """Removes the edge between node u and node v. Returns True if removed."""
        if u not in self.adj_list or v not in self.adj_list:
            return False

        original_len = len(self.adj_list[u])
        self.adj_list[u] = [(neighbor, edge_weight) for neighbor, edge_weight in self.adj_list[u] if neighbor != v]
        removed = len(self.adj_list[u]) < original_len

        if not self.is_directed and v in self.adj_list:
            self.adj_list[v] = [(neighbor, edge_weight) for neighbor, edge_weight in self.adj_list[v] if neighbor != u]

        if removed:
            self._rebuild_matrix()
        return removed

    def remove_node(self, node: str) -> bool:
        """Removes a node and all connecting edges. Returns True if removed."""
        if node not in self.nodes:
            return False

        self.nodes.remove(node)
        self.adj_list.pop(node, None)

        for current_node in self.adj_list:
            self.adj_list[current_node] = [
                (neighbor, edge_weight) for neighbor, edge_weight in self.adj_list[current_node] if neighbor != node
            ]

        self._rebuild_matrix()
        return True

    def display_structure(self) -> None:
        """Prints a human-readable summary of the graph topology and linkages."""
        print("\n" + "=" * 50)
        print(f"CURRENT GRAPH CONFIGURATION ({'DIRECTED' if self.is_directed else 'UNDIRECTED'})")
        print("=" * 50)
        print(f"Total Vertices (Nodes): {len(self.nodes)}")

        total_edges = sum(len(edges) for edges in self.adj_list.values())
        if not self.is_directed:
            total_edges //= 2
        print(f"Total Edges (Links):    {total_edges}")
        print("-" * 50)

        if not self.nodes:
            print("Graph layout is currently completely empty.")
            print("=" * 50)
            return

        print("Adjacency List View:")
        for node in sorted(self.nodes):
            edges = self.adj_list.get(node, [])
            if edges:
                conn_strings = [f"{neighbor} (w:{weight})" for neighbor, weight in edges]
                connections_line = ", ".join(conn_strings)
            else:
                connections_line = "Isolated (No Edges)"
            print(f"Node [{node}] -> {connections_line}")

        print("\nAdjacency Matrix View:")
        print(self.ascii_matrix())
        print("=" * 50 + "\n")

    def ascii_matrix(self) -> str:
        """Produces a compact ASCII adjacency matrix for visual inspection."""
        if not self.node_order:
            return "Graph is empty."

        labels = self.node_order
        width = max(len(label) for label in labels) + 2
        header = " " * (width + 1) + " ".join(f"{label:>{width}}" for label in labels)
        rows = [header]

        for row_node in labels:
            row_values = [f"{row_node:>{width}}"]
            row_idx = self.index_map[row_node]
            for col_node in labels:
                col_idx = self.index_map[col_node]
                value = self.matrix[row_idx][col_idx]
                cell = "0" if value is None else str(int(value) if value.is_integer() else value)
                row_values.append(f"{cell:>{width}}")
            rows.append(" ".join(row_values))
        return "\n".join(rows)

    def visualize_traversal(self, start_node: str, algorithm: str = "BFS") -> None:
        """Prints a step-by-step traversal animation using queue/stack states."""
        if start_node not in self.nodes:
            print(f"Error: Node '{start_node}' does not exist.")
            return

        if algorithm.upper() == "BFS":
            visited = {start_node}
            queue = deque([start_node])
            order = []
            print(f"\n=== BFS VISUALIZATION from {start_node} ===")
            print(f"Initial queue: {list(queue)}")
            while queue:
                current = queue.popleft()
                order.append(current)
                print(f"Visit: {current}")
                neighbors = sorted(self.adj_list.get(current, []), key=lambda item: item[0])
                print(f"  Neighbors: {[neighbor for neighbor, _ in neighbors]}")
                for neighbor, weight in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        print(f"  Enqueue {neighbor} via {current} (weight {weight})")
                print(f"  Queue now: {list(queue)}")
            print(f"Final BFS order: {order}\n")
        elif algorithm.upper() == "DFS":
            visited = set()
            stack = [start_node]
            order = []
            print(f"\n=== DFS VISUALIZATION from {start_node} ===")
            print(f"Initial stack: {stack}")
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                order.append(current)
                print(f"Visit: {current}")
                neighbors = sorted(self.adj_list.get(current, []), key=lambda item: item[0], reverse=True)
                print(f"  Neighbors: {[neighbor for neighbor, _ in neighbors]}")
                for neighbor, weight in neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)
                        print(f"  Push {neighbor} via {current} (weight {weight})")
                print(f"  Stack now: {stack}")
            print(f"Final DFS order: {order}\n")
        else:
            print(f"Unsupported algorithm: {algorithm}. Use BFS or DFS.")

    def breadth_first_search(self, start_node: str) -> List[str]:
        if start_node not in self.nodes:
            return []
        visited, queue, order = {start_node}, deque([start_node]), []
        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor, _ in self.adj_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order

    def depth_first_search(self, start_node: str) -> List[str]:
        if start_node not in self.nodes:
            return []
        visited, stack, order = set(), [start_node], []
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                order.append(current)
                for neighbor, _ in reversed(self.adj_list.get(current, [])):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return order

    def dijkstra(self, start_node: str) -> Dict[str, float]:
        if start_node not in self.nodes:
            return {}
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0.0
        priority_queue = [(0.0, start_node)]

        while priority_queue:
            curr_dist, curr_node = heapq.heappop(priority_queue)
            if curr_dist > distances[curr_node]:
                continue
            for neighbor, weight in self.adj_list.get(curr_node, []):
                distance = curr_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        return distances

    def compare_performance(self, node_count: int = 80, density: float = 0.20) -> str:
        """Compares runtime characteristics of adjacency-list vs adjacency-matrix operations."""
        if node_count < 2:
            return "At least 2 nodes are needed for a meaningful comparison."
        if not 0 < density <= 1:
            return "Density must be a value in the range (0, 1]."

        benchmark = Graph(is_directed=False)
        for i in range(node_count):
            benchmark.add_node(f"N{i}")

        nodes = benchmark.node_order
        edge_step = max(1, int(round(1 / density)))
        for i in range(node_count):
            for j in range(i + 1, node_count):
                if (i * 7 + j * 11) % edge_step == 0:
                    benchmark.add_edge(nodes[i], nodes[j], float((i + j) % 9 + 1))

        target_u = nodes[0]
        target_v = nodes[1]

        iterations = 2000

        list_start = time.perf_counter()
        for _ in range(iterations):
            adjacency = benchmark.adj_list.get(target_u, [])
            _ = any(neighbor == target_v for neighbor, _ in adjacency)
        list_edge_check = time.perf_counter() - list_start

        matrix_start = time.perf_counter()
        for _ in range(iterations):
            u_index = benchmark.index_map[target_u]
            v_index = benchmark.index_map[target_v]
            _ = benchmark.matrix[u_index][v_index] is not None
        matrix_edge_check = time.perf_counter() - matrix_start

        list_start = time.perf_counter()
        for _ in range(50):
            for node in nodes:
                for neighbor, _ in benchmark.adj_list.get(node, []):
                    _ = neighbor
        list_traversal = time.perf_counter() - list_start

        matrix_start = time.perf_counter()
        for _ in range(50):
            for row in benchmark.matrix:
                for value in row:
                    _ = value
        matrix_traversal = time.perf_counter() - matrix_start

        list_faster_edge = list_edge_check < matrix_edge_check
        list_faster_traversal = list_traversal < matrix_traversal

        return (
            "Performance Comparison Report\n"
            f"- Nodes: {node_count}\n"
            f"- Density: {density:.2f}\n"
            "- Edge existence check:\n"
            f"  * Adjacency List: {list_edge_check:.6f} seconds\n"
            f"  * Adjacency Matrix: {matrix_edge_check:.6f} seconds\n"
            "- Full-neighbor iteration:\n"
            f"  * Adjacency List: {list_traversal:.6f} seconds\n"
            f"  * Adjacency Matrix: {matrix_traversal:.6f} seconds\n"
            "- Interpretation:\n"
            f"  * Adjacency List is {'faster' if list_faster_edge else 'slower'} for direct edge checks in this test.\n"
            f"  * Adjacency List is {'faster' if list_faster_traversal else 'slower'} for neighbor traversal in this test.\n"
            "  * In sparse graphs, adjacency lists usually win on memory and traversal efficiency.\n"
            "  * In dense graphs, adjacency matrices often win for constant-time edge lookup."
        )


# ==========================================
# INTERACTIVE TERMINAL LOOP INTERFACE
# ==========================================
def run_graph_interactive_menu():
    print("Welcome to the Graph System Deployment Module!")
    direction_input = input("Should the graph be directed? (y/N): ").strip().lower()
    is_directed = direction_input == 'y' or direction_input == 'yes'
    
    g = Graph(is_directed=is_directed)
    
    # Pre-populate sample nodes for instant validation ease
    g.add_edge("A", "B", 4)
    g.add_edge("B", "C", 2)
    print("\n[System Notice]: Pre-populated layout with initial sample nodes (A -> B -> C).")

    while True:
        print("\n--- GRAPH MANAGEMENT SYSTEM MENU ---")
        print("1. Add Node")
        print("2. Add/Update Edge")
        print("3. Remove Node")
        print("4. Remove Edge")
        print("5. Display Graph Structure & Connections")
        print("6. Run Traversal Algorithms (BFS/DFS)")
        print("7. Run Shortest Path (Dijkstra)")
        print("8. Compare Adjacency List vs. Adjacency Matrix")
        print("9. Visualize Graph Traversal")
        print("10. Exit Application")
        
        choice = input("\nSelect an operational flag (1-10): ").strip()
        
        if choice == "1":
            node = input("Enter unique Node label name: ").strip()
            if g.add_node(node):
                print(f"Success: Node '{node}' added.")
            else:
                print(f"Notice: Node '{node}' already exists or name was invalid.")
                
        elif choice == "2":
            u = input("Enter Source Node (u): ").strip()
            v = input("Enter Destination Node (v): ").strip()
            try:
                weight_input = input("Enter Edge Weight cost (Default 1.0): ").strip()
                weight = float(weight_input) if weight_input else 1.0
                g.add_edge(u, v, weight)
                print(f"Success: Created edge link ({u} -> {v}) with cost {weight}.")
            except ValueError:
                print("Error: Weight parsing failed. Enter a valid numeric float structure.")
                
        elif choice == "3":
            node = input("Enter Node label to purge: ").strip()
            if g.remove_node(node):
                print(f"Success: Node '{node}' and all its connections dropped completely.")
            else:
                print(f"Error: Node '{node}' not found.")
                
        elif choice == "4":
            u = input("Enter source vertex: ").strip()
            v = input("Enter target vertex: ").strip()
            if g.remove_edge(u, v):
                print(f"Success: Edge boundary between {u} and {v} closed.")
            else:
                print("Error: Targeted edge combination link path not found.")
                
        elif choice == "5":
            g.display_structure()
            
        elif choice == "6":
            start = input("Enter entry-point root node for traversal exploration: ").strip()
            if start not in g.nodes:
                print("Error: Target vertex doesn't exist within layout registry.")
                continue
            g.visualize_traversal(start, "BFS")
            g.visualize_traversal(start, "DFS")
            print(f"BFS Order Sequence Output: {g.breadth_first_search(start)}")
            print(f"DFS Order Sequence Output: {g.depth_first_search(start)}")
            
        elif choice == "7":
            start = input("Enter source node context for Dijkstra mapping: ").strip()
            if start not in g.nodes:
                print("Error: Specified location node not found.")
                continue
            paths = g.dijkstra(start)
            print(f"\nShortest Paths relative cost matrix from '{start}':")
            for target, cost in paths.items():
                cost_str = "Unreachable" if cost == float('inf') else str(cost)
                print(f"  * Path tracking target [{target}] -> Cost Metric: {cost_str}")
                
        elif choice == "8":
            try:
                node_count = int(input("Enter node count for comparison benchmark (default 80): ").strip() or "80")
                density = float(input("Enter graph density between 0 and 1 (default 0.20): ").strip() or "0.20")
                print("\n" + g.compare_performance(node_count=node_count, density=density))
            except ValueError:
                print("Error: Enter valid numeric values for node count and density.")

        elif choice == "9":
            start = input("Enter entry-point root node for visual traversal: ").strip()
            if start not in g.nodes:
                print("Error: Target vertex doesn't exist within layout registry.")
                continue
            g.display_structure()
            algorithm = input("Select traversal style (BFS or DFS): ").strip().upper()
            g.visualize_traversal(start, algorithm)

        elif choice == "10":
            print("\nShutting down Graph Framework environment module. Goodbye!")
            break
        else:
            print("Invalid flag chosen. Please specify a sequence number integer (1-10).")



def non_linear_data_structure_implementation():
    """Run the interactive menu for non-linear data structure experiments."""
    tree_demo = TreeDemoInterface()
    tree_demo.run()

    # Run the graph demo after the tree demo
    run_graph_interactive_menu()

# ================================================================
# END - NON-LINEAR DATA STRUCTURE IMPLEMENTATION MODULE
# ================================================================





# ==========================================
#  SORTING ALGORITHMS IMPLEMENTATION
# ==========================================

def bubble_sort(arr):
    """Sort a sequence with bubble sort and return a new list.

    Bubble sort repeatedly swaps adjacent out-of-order values. Its worst-case
    time complexity is O(n^2), while the early-exit check gives it an O(n)
    best case for an already sorted input.
    """
    n = len(arr)
    arr_copy = list(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                swapped = True
        if not swapped:
            break
    return arr_copy

def selection_sort(arr):
    """Sort a sequence with selection sort and return a new list.

    On each pass, the smallest remaining value is selected and placed at the
    next output position. The algorithm takes O(n^2) time in all input cases.
    """
    n = len(arr)
    arr_copy = list(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr_copy[j] < arr_copy[min_idx]:
                min_idx = j
        arr_copy[i], arr_copy[min_idx] = arr_copy[min_idx], arr_copy[i]
    return arr_copy

def insertion_sort(arr):
    """Sort a sequence with insertion sort and return a new list.

    Values are inserted into the sorted prefix on the left. This takes O(n)
    time for sorted input and O(n^2) time in the worst case.
    """
    arr_copy = list(arr)
    for i in range(1, len(arr_copy)):
        key = arr_copy[i]
        j = i - 1
        while j >= 0 and key < arr_copy[j]:
            arr_copy[j + 1] = arr_copy[j]
            j -= 1
        arr_copy[j + 1] = key
    return arr_copy

def merge_sort(arr):
    """
    Sort a sequence with bottom-up merge sort and return a new list.

    The iterative implementation avoids recursion limits and uses a shared
    workspace during merges. It runs in O(n log n) time and uses O(n) space.
    """
    arr_copy = list(arr)
    low = 0
    high = len(arr_copy) - 1
    
    # Work space array to avoid rebuilding arrays on every merge step
    temp = [0] * len(arr_copy)
    
    width = 1
    while width <= high - low:
        for i in range(low, high + 1, 2 * width):
            # Define boundaries for sub-arrays
            left = i
            mid = i + width - 1
            right = min(i + 2 * width - 1, high)
            
            # Merge subarrays in place using the temp buffer
            if mid < right:
                _merge(arr_copy, temp, left, mid, right)
        width *= 2
        
    return arr_copy

def _merge(arr, temp, left, mid, right):
    """Merge two adjacent sorted ranges into ``arr`` using ``temp``."""
    i = left     # Index for left subarray
    j = mid + 1  # Index for right subarray
    k = left     # Index for temp result subarray
    
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1
        
    while i <= mid:
        temp[k] = arr[i]
        i += 1
        k += 1
        
    while j <= right:
        temp[k] = arr[j]
        j += 1
        k += 1
        
    # Copy sorted items back to original tracking structure
    for i in range(left, right + 1):
        arr[i] = temp[i]


def _partition(arr, low, high):
    """Partition an inclusive range using Hoare's scheme.

    The pivot is the median of the first, middle, and last values, which helps
    avoid highly unbalanced partitions for already sorted or reverse-sorted
    input. Return the final index of the left partition; both partitions
    contain at least one element when ``low < high``.
    """
    middle = (low + high) // 2
    pivot = sorted((arr[low], arr[middle], arr[high]))[1]
    left = low
    right = high

    while True:
        while arr[left] < pivot:
            left += 1
        while arr[right] > pivot:
            right -= 1
        if left >= right:
            return right
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1


def quick_sort(arr):
    """Sort a sequence with quick sort and return a new list.

    Quick sort uses a divide-and-conquer strategy to partition the input
    around a pivot. It has an average time complexity of O(n log n), O(n^2)
    worst-case time, and O(n) output space. The implementation uses an explicit
    range stack to avoid Python recursion-depth failures on large inputs.
    """
    arr_copy = list(arr)
    if len(arr_copy) < 2:
        return arr_copy

    pending = [(0, len(arr_copy) - 1)]
    while pending:
        low, high = pending.pop()
        while low < high:
            split = _partition(arr_copy, low, high)
            left_range = (low, split)
            right_range = (split + 1, high)

            # Process the smaller side immediately and defer the larger side.
            # This keeps the explicit stack bounded even for skewed partitions.
            if split - low < high - split:
                pending.append(right_range)
                low, high = left_range
            else:
                pending.append(left_range)
                low, high = right_range
    return arr_copy


def bubble_sort_visual(arr, silent=False):
    """
    Sort a list of integers in ascending order and visualize every comparison.

    The input is copied, so the caller's list is not modified. During each pass,
    adjacent values being compared are marked with asterisks; the output states
    whether the pair stays in place or swaps. A pass with no swaps ends the
    algorithm early.

    Args:
        arr: A list containing only integers (booleans are not accepted).
        silent: If True, suppress visualization output. This is intended for
            benchmarks; the returned sorted result is unchanged.

    Returns:
        A new list containing the values in ascending order.

    Raises:
        TypeError: If arr is not a list of integers.

    Complexity:
        Worst/average time O(n^2); best-case time O(n) for already sorted data;
        auxiliary space O(n) for the non-mutating working copy.
    """
    if not isinstance(arr, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in arr
    ):
        raise TypeError("bubble_sort_visual expects a list of integers")

    working_arr = arr.copy()
    n = len(working_arr)
    if not silent:
        print(f"\n[START] Bubble Sort Initialization: {working_arr}")
    
    for i in range(n):
        swapped = False
        if not silent:
            print(f"--- Pass {i + 1} ---")
        for j in range(0, n - i - 1):
            if not silent:
                state = [str(x) for x in working_arr]
                state[j] = f"*{state[j]}*"
                state[j+1] = f"*{state[j+1]}*"
            
            if working_arr[j] > working_arr[j+1]:
                if not silent:
                    print(f"Comparing: {' '.join(state)} | {working_arr[j]} > {working_arr[j+1]} -> Swapping!")
                working_arr[j], working_arr[j+1] = working_arr[j+1], working_arr[j]
                swapped = True
            else:
                if not silent:
                    print(f"Comparing: {' '.join(state)} | {working_arr[j]} <= {working_arr[j+1]} -> Position OK")
                
        if not swapped:
            if not silent:
                print("[EARLY EXIT] Array is fully sorted. No swaps were made during this pass.")
            break
            
    if not silent:
        print(f"[FINISHED] Sorted Array Result: {working_arr}\n")
    return working_arr


def _parse_integer_list(raw_values):
    """Parse comma- or whitespace-separated integers entered at the prompt."""
    try:
        return [int(value) for value in raw_values.replace(",", " ").split()]
    except ValueError as error:
        raise ValueError("Enter whole numbers separated by commas or spaces.") from error


def selection_sort_visual(arr, silent=False):
    """
    Performs Selection Sort, visualizing how the minimum element is collected.
    Maintains rigorous O(N^2) operational profiles across all states.
    """
    working_arr = arr.copy()
    n = len(working_arr)
    if not silent:
        print(f"\n[START] Selection Sort Initialization: {working_arr}")
    
    for i in range(n):
        min_idx = i
        if not silent:
            print(f"--- Step {i + 1}: Setting current minimum candidate index to {i} (Value: {working_arr[i]}) ---")
        
        for j in range(i + 1, n):
            if working_arr[j] < working_arr[min_idx]:
                if not silent:
                    print(f"   Found smaller element! Index {j} (Value: {working_arr[j]}) < Index {min_idx} (Value: {working_arr[min_idx]})")
                min_idx = j
                
        if min_idx != i:
            if not silent:
                print(f"Swapping starting position index {i} ({working_arr[i]}) with absolute minimum index {min_idx} ({working_arr[min_idx]})")
            working_arr[i], working_arr[min_idx] = working_arr[min_idx], working_arr[i]
        else:
            if not silent:
                print(f"Index {i} already contains the absolute minimum value for this sub-array.")
            
        if not silent:
            print(f"Current array state: {working_arr}")
        
    if not silent:
        print(f"[FINISHED] Sorted Array Result: {working_arr}\n")
    return working_arr


def insertion_sort_visual(arr, silent=False):
    """
    Performs Insertion Sort, visualizing how elements are inserted into the sorted prefix.
    Maintains O(N^2) operational profiles across all states, with O(N) best-case for sorted input.
    """
    working_arr = arr.copy()
    n = len(working_arr)
    if not silent:
        print(f"\n[START] Insertion Sort Initialization: {working_arr}")
    
    for i in range(1, n):
        key = working_arr[i]
        j = i - 1
        if not silent:
            print(f"--- Step {i}: Inserting value {key} at index {i} into sorted prefix ---")
        
        while j >= 0 and working_arr[j] > key:
            if not silent:
                print(f"   {working_arr[j]} (index {j}) > {key} -> Shifting {working_arr[j]} to index {j + 1}")
            working_arr[j + 1] = working_arr[j]
            j -= 1
            
        working_arr[j + 1] = key
        if not silent:
            print(f"Inserted {key} at index {j + 1}. Current array state: {working_arr}")
        
    if not silent:
        print(f"[FINISHED] Sorted Array Result: {working_arr}\n")
    return working_arr



def merge_sort_visual(arr, silent=False):
    """
    Performs Merge Sort, visualizing how sub-arrays are merged.
    Maintains O(N log N) operational profiles across all states.
    """
    working_arr = arr.copy()
    if not silent:
        print(f"\n[START] Merge Sort Initialization: {working_arr}")
    
    def _merge_sort_recursive(arr, left, right):
        if left < right:
            mid = (left + right) // 2
            _merge_sort_recursive(arr, left, mid)
            _merge_sort_recursive(arr, mid + 1, right)
            _merge(arr, left, mid, right)
            if not silent:
                print(f"Merged sub-arrays from index {left} to {right}: {arr[left:right+1]}")
    
    def _merge(arr, left, mid, right):
        L = arr[left:mid + 1]
        R = arr[mid + 1:right + 1]
        i = j = 0
        k = left
        
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    
    _merge_sort_recursive(working_arr, 0, len(working_arr) - 1)
    
    if not silent:
        print(f"[FINISHED] Sorted Array Result: {working_arr}\n")
    return working_arr


def quick_sort_visual(arr, silent=False):
    """
    Performs Quick Sort, visualizing how the array is partitioned around pivots.
    Maintains O(N log N) average-case operational profiles, with O(N^2) worst-case.
    """
    working_arr = arr.copy()
    if not silent:
        print(f"\n[START] Quick Sort Initialization: {working_arr}")
    
    def _quick_sort_recursive(arr, low, high):
        if low < high:
            pivot_index = _partition(arr, low, high)
            if not silent:
                print(f"Partitioned with pivot at index {pivot_index}: {arr[low:high+1]}")
            _quick_sort_recursive(arr, low, pivot_index)
            _quick_sort_recursive(arr, pivot_index + 1, high)
    
    def _partition(arr, low, high):
        pivot = arr[(low + high) // 2]
        i = low - 1
        j = high + 1
        
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                return j
            arr[i], arr[j] = arr[j], arr[i]
            if not silent:
                print(f"Swapped indices {i} and {j}: {arr[low:high+1]}")
    
    _quick_sort_recursive(working_arr, 0, len(working_arr) - 1)
    
    if not silent:
        print(f"[FINISHED] Sorted Array Result: {working_arr}\n")
    return working_arr


def run_sorting_visualization():
    """Run the sorting visualization interface."""
    print("Welcome to the Sorting Algorithm Visualization Module!")
    print("You can input a list of integers (comma or space-separated) to see how different sorting algorithms work step-by-step.\n")
    
    raw_input = input("Enter a list of integers to sort: ").strip()
    try:
        values = _parse_integer_list(raw_input)
    except ValueError as e:
        print(f"Input Error: {e}")
        return
    
    if not values:
        print("No values provided. Exiting visualization.")
        return
    
    print("\nSelect a sorting algorithm to visualize:")
    print("1. Bubble Sort")
    print("2. Selection Sort")
    print("3. Insertion Sort")
    print("4. Merge Sort")
    print("5. Quick Sort")
    print("0. Exit Sorting Visualization Demo")
    
    choice = input("Enter the number corresponding to your choice: ").strip()
    
    if choice == "1":
        bubble_sort_visual(values)
    elif choice == "2":
        selection_sort_visual(values)
    elif choice == "3":
        insertion_sort_visual(values)
    elif choice == "4":
        merge_sort_visual(values)
    elif choice == "5":
        quick_sort_visual(values)
    elif choice == "0":
        print("Exiting Sorting Visualization Demo.")
        return
    else:
        print("Invalid choice. Please select a number between 1 and 5.")


# ==========================================
# SORTING ALGORITHM DATASET GENERATOR
# ==========================================

def generate_dataset(data_type, size):
    """Create a dataset with the requested ordering and number of values.

    Supported types are ``Random``, ``Already Sorted``, ``Reverse Sorted``,
    and ``Partially Sorted``. ``size`` must be non-negative. A ``ValueError``
    is raised for an unknown type or a negative size.
    """
    if size < 0:
        raise ValueError("Dataset size cannot be negative")
    if size == 0:
        return []

    if data_type == "Random":
        return [random.randint(1, size * 10) for _ in range(size)]
    elif data_type == "Already Sorted":
        return list(range(1, size + 1))
    elif data_type == "Reverse Sorted":
        return list(range(size, 0, -1))
    elif data_type == "Partially Sorted":
        data = list(range(1, size + 1))
        swaps = max(1, int(size * 0.05))
        for _ in range(swaps):
            idx1 = random.randint(0, size - 1)
            idx2 = random.randint(0, size - 1)
            data[idx1], data[idx2] = data[idx2], data[idx1]
        return data
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def format_array_preview(values, preview_size=5):
    """Return a readable full array or bounded preview for display."""
    if len(values) <= preview_size * 2:
        return str(values)
    return f"{values[:preview_size]} ... {values[-preview_size:]} (length={len(values)})"


# ==========================================
# SORTING ALGORITHM PERFORMANCE TIMING HARNESS
# ==========================================

def evaluate_performance():
    """Run each sorting algorithm and return the collected timings.

    Returns:
        tuple: ``(results, dataset_types, sizes)``. ``results`` is keyed by
        ``(dataset_type, size)`` and then by algorithm name. Every algorithm
        runs for every dataset combination; ``TIME_LIMIT`` only produces a
        warning when a run takes longer than the advisory threshold.
    """
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort
    }
    
    dataset_types = ["Random", "Already Sorted", "Reverse Sorted", "Partially Sorted"]
    sizes = [1000, 5000, 10000] #50000 takes a very long time to run  # FIXED: Defined values correctly
    results = {}
    before_previews = []
    after_previews = []
    
    # This threshold reports slow runs but does not skip any combinations.
    TIME_LIMIT = 180.0
    
    print("⏳ Executing benchmarking suite... (All operations will run)\n")
    
    for size in sizes:
        for d_type in dataset_types:
            raw_data = generate_dataset(d_type, size)
            before_previews.append(
                (d_type, size, format_array_preview(raw_data))
            )
            
            if (d_type, size) not in results:
                results[(d_type, size)] = {}
                
            for algo_name, algo_func in algorithms.items():
                start_time = time.perf_counter()
                sorted_data = algo_func(raw_data)
                end_time = time.perf_counter()
                
                duration = end_time - start_time
                results[(d_type, size)][algo_name] = duration
                after_previews.append(
                    (d_type, size, algo_name, format_array_preview(sorted_data))
                )
                
                if duration > TIME_LIMIT:
                    print(f"⚠️  {algo_name} exceeded the advisory limit of {TIME_LIMIT}s on {d_type} (Size: {size}). Continuing benchmark.")

    print("\n" + "=" * 75)
    print("BEFORE SORTING DATASETS")
    print("=" * 75)
    displayed_size = None
    for d_type, size, preview in before_previews:
        if size != displayed_size:
            print(f"\n--- Processing Dataset Size: {size} elements ---")
            displayed_size = size
        print(f"{d_type} dataset (size={size}):")
        print(f"  {preview}")

    print("\n" + "=" * 75)
    print("AFTER SORTING DATASETS")
    print("=" * 75)
    displayed_size = None
    for d_type, size, algo_name, preview in after_previews:
        if size != displayed_size:
            print(f"\n--- Processing Dataset Size: {size} elements ---")
            displayed_size = size
        print(f"{d_type} dataset (size={size}) - {algo_name}:")
        print(f"  {preview}")
    return results, dataset_types, sizes

# ==========================================
# SORTING REPORT GENERATOR INTERFACE
# ==========================================

def display_sorting_report(results, dataset_types, sizes):
    """Print timing tables and the fastest algorithm for each dataset size."""
    print("\n" + "="*75)
    print("                📊 SORTING PERFORMANCE BENCHMARK REPORT               ")
    print("="*75)
    
    for d_type in dataset_types:
        print(f"\n🔹 DATASET TYPE: {d_type.upper()}")
        print("-" * 75)
        print(f"{'Size':<10} | {'Bubble':<12} | {'Selection':<12} | {'Insertion':<12} | {'Merge':<12} | {'Quick':<12}")
        print("-" * 75)
        
        winners = {}
        
        for size in sizes:
            row_times = results[(d_type, size)]
            display_vals = {}
            valid_times = {}
            
            for name, t in row_times.items():
                if t == float('inf'):
                    display_vals[name] = "TIMEOUT"
                else:
                    display_vals[name] = f"{t:.4f}s"
                    valid_times[name] = t
            
            best_algo = min(valid_times, key=valid_times.get) if valid_times else "None"
            winners[size] = best_algo
            
            print(f"{size:<10} | {display_vals['Bubble Sort']:<12} | {display_vals['Selection Sort']:<12} | {display_vals['Insertion Sort']:<12} | {display_vals['Merge Sort']:<12} | {display_vals['Quick Sort']:<12}")
        
        print("-" * 75)
        print("💡 Efficiency Leaders:")
        for size, winner in winners.items():
            print(f"  Size {size:<5}: {winner} performed best")
    # -----------------------------------------------------------------
    # PART B: SORTING ALGORITHM BENCHMARKS
    # -----------------------------------------------------------------
    sort_sizes = [100, 1000, 10000] # 20000]
    print("\n" + "-"*75)
    print("                 SORTING RUNTIME BENCHMARK SUITE")
    print("-" * 75)
    print(f" {'Dataset Size':<12} | {'Bubble Sort (ms)':<25} | {'Selection Sort (ms)':<25} | {'Insertion Sort (ms)':<25} | {'Merge Sort (ms)':<25} | {'Quick Sort (ms)':<25}")
    print("-" * 75)
    benchmark_runs = 5  # Number of runs to average for timing
    for size in sort_sizes:
        unsorted_base = [random.randint(1, 10000) for _ in range(size)]
        
        bubble_elapsed_ns = 0
        selection_elapsed_ns = 0
        insertion_elapsed_ns = 0
        merge_elapsed_ns = 0
        quick_elapsed_ns = 0
        for _ in range(benchmark_runs):
            bubble_sample = unsorted_base.copy()
            t0 = time.perf_counter_ns()
            bubble_sort_visual(bubble_sample, silent=True)
            t1 = time.perf_counter_ns()
            bubble_elapsed_ns += t1 - t0

            selection_sample = unsorted_base.copy()
            t2 = time.perf_counter_ns()
            selection_sort_visual(selection_sample, silent=True)
            t3 = time.perf_counter_ns()
            selection_elapsed_ns += t3 - t2

            insertion_sample = unsorted_base.copy()
            t4 = time.perf_counter_ns()
            insertion_sort_visual(insertion_sample, silent=True)
            t5 = time.perf_counter_ns()
            insertion_elapsed_ns += t5 - t4

            merge_sample = unsorted_base.copy()
            t6 = time.perf_counter_ns()
            merge_sort_visual(merge_sample, silent=True)
            t7 = time.perf_counter_ns()
            merge_elapsed_ns += t7 - t6

            quick_sample = unsorted_base.copy()
            t8 = time.perf_counter_ns()
            quick_sort_visual(quick_sample, silent=True)
            t9 = time.perf_counter_ns()
            quick_elapsed_ns += t9 - t8

        bubble_time_ms = bubble_elapsed_ns / benchmark_runs / 1_000_000.0
        selection_time_ms = selection_elapsed_ns / benchmark_runs / 1_000_000.0
        insertion_time_ms = insertion_elapsed_ns / benchmark_runs / 1_000_000.0
        merge_time_ms = merge_elapsed_ns / benchmark_runs / 1_000_000.0
        quick_time_ms = quick_elapsed_ns / benchmark_runs / 1_000_000.0

        print(f" {size:<12} | {bubble_time_ms:<25.4f} | {selection_time_ms:<25.4f} | {insertion_time_ms:<25.4f} | {merge_time_ms:<25.4f} | {quick_time_ms:<25.4f}")
        
        print("="*75 + "\n")

#def run_sorting_performace_demo():
    


def create_performance_charts(results, dataset_types, sizes, output_dir="charts"):
    """Save one execution-time line chart for each dataset type.

    A logarithmic y-axis is used so that differences between fast and slow
    algorithms remain visible.

    Returns:
        list[Path]: Paths to the generated PNG files.
    """

    algorithms = ["Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort", "Quick Sort"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    for dataset_type in dataset_types:
        figure, axis = plt.subplots(figsize=(10, 6))

        for algorithm in algorithms:
            times = []
            for size in sizes:
                duration = results[(dataset_type, size)][algorithm]
                times.append(duration if duration != float("inf") else None)
            axis.plot(sizes, times, marker="o", linewidth=2, label=algorithm)

        axis.set_title(f"Sorting Performance: {dataset_type}")
        axis.set_xlabel("Dataset size (elements)")
        axis.set_ylabel("Execution time (seconds)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.4)
        axis.legend()
        figure.tight_layout()

        filename = dataset_type.lower().replace(" ", "_") + "_performance.png"
        chart_path = output_path / filename
        figure.savefig(chart_path, dpi=150)
        plt.close(figure)
        chart_paths.append(chart_path)

    return chart_paths


def run_sorting_demo():
    """Run the sorting benchmark and display results with charts."""
    while True:
        print("\n===========================================================================================")
        print("    Sorting Algorithm Performance Benchmark Suite")
        print("===========================================================================================")
        print("1. Generate Sorting Dataset")
        print("2. Sorting Algorithms Performance (Bubble, Selection, Insertion, Merge, Quick)")
        print("3. Display Sorting Report")
        print("4. Run Sorting Visualization ")
        print("5. Generate Performance Charts")
        print("0. Exit Sorting Demo")
        
        main_choice = input("Select an option: ").strip()
        
        if main_choice == "0":
            print("Exiting the Sorting Demo. Goodbye!")
            return
        elif main_choice == "1":
            print("\nGenerating Sorting Dataset...")
            try:
                item_count = int(input("Enter the number of items for sorting dataset (e.g., 1000): "))
                if item_count <= 0:
                    print("Please enter a positive integer.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
                continue
            dataset_formats = ["Random", "Already Sorted", "Reverse Sorted", "Partially Sorted"]
            for d_type in dataset_formats:
                dataset = generate_dataset(d_type, item_count)
                print(f"\n{d_type} dataset (size={item_count}):")
                print(f"  {format_array_preview(dataset)}")
        elif main_choice == "2":
            print("\nRunning Sorting Algorithms Demo...")
            evaluate_performance()
        elif main_choice == "3":
            print("\nDisplaying Sorting Report...")
            raw_results, dataset_formats, data_scales = evaluate_performance()
            display_sorting_report(raw_results, dataset_formats, data_scales)
        elif main_choice == "4":
            print("\nRunning Sorting Visualization...")
            run_sorting_visualization()
        elif main_choice == "5":
            print("\nGenerating Performance Charts...")
            raw_results, dataset_formats, data_scales = evaluate_performance()
            chart_files = create_performance_charts(raw_results, dataset_formats, data_scales)
            print(f"\n📈 PerformanceCharts saved to: {', '.join(str(path) for path in chart_files)}")
        else:
            print("Invalid choice. Please select a valid option 0-5.")
            
    


# ==========================================
# END - SORTING ALGORITHM IMPLEMENTATION MODULE
# ==========================================


# ==========================================
#  SEARCH ALGORITHMS IMPLEMENTATION
# ==========================================

def linear_search_visual(arr, target):
    """
    Performs a linear search, printing a text visualization of its progress.
    Exhibits O(N) worst-case scaling over unsorted raw array structures.
    """
    print(f"\n[START] Linear Search for target: {target}")
    print(f"Array status: {arr}")
    
    for index, element in enumerate(arr):
        pointer_line = ["   "] * len(arr)
        pointer_line[index] = " ^ "
        print(f"Checking index {index:<2} -> Value: {element:<3} | " + "".join(pointer_line))
        
        if element == target:
            print(f"[FOUND] Success! Target {target} discovered at index {index}.\n")
            return index
            
    print(f"[NOT FOUND] Target {target} is missing from the array.\n")
    return -1


def binary_search_visual(arr, target):
    """
    Performs a binary search on a sorted array, visualizing bounds changes.
    Exhibits O(log N) scaling via divide-and-conquer domain reduction.
    """
    print(f"\n[START] Binary Search for target: {target}")
    print(f"Sorted array: {arr}")
    
    low = 0
    high = len(arr) - 1
    step = 1

    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]
        
        visual_track = []
        for i, value in enumerate(arr):
            if i == mid:
                visual_track.append(f"[{value}]")
            elif i == low:
                visual_track.append(f"L({value})")
            elif i == high:
                visual_track.append(f"H({value})")
            else:
                visual_track.append(str(value))
        
        print(f"Step {step:<2} | Low: {low} (Val:{arr[low]}), Mid: {mid} (Val:{guess}), High: {high} (Val:{arr[high]})")
        print("        Track View: " + " -> ".join(visual_track))
        
        if guess == target:
            print(f"[FOUND] Success! Target {target} discovered at index {mid}.\n")
            return mid
        if guess > target:
            print(f"        -> {guess} > {target}. Target must be in the LOWER half. Shifting High bound.\n")
            high = mid - 1
        else:
            print(f"        -> {guess} < {target}. Target must be in the UPPER half. Shifting Low bound.\n")
            low = mid + 1
        step += 1
            
    print(f"[NOT FOUND] Target {target} is missing from the array.\n")
    return -1



def quickselect(arr, k):
    """
    Quickselect algorithm to find the k-th smallest element in an unsorted array.
    This is a selection algorithm with average O(N) time complexity.
    """
    if k < 1 or k > len(arr):
        raise ValueError("k must be between 1 and the length of the array.")
    
    def partition(left, right, pivot_index):
        pivot_value = arr[pivot_index]
        # Move pivot to end
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
        store_index = left
        for i in range(left, right):
            if arr[i] < pivot_value:
                arr[store_index], arr[i] = arr[i], arr[store_index]
                store_index += 1
        # Move pivot to its final place
        arr[right], arr[store_index] = arr[store_index], arr[right]
        return store_index

    left, right = 0, len(arr) - 1
    while True:
        if left == right:
            return arr[left]
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        if k - 1 == pivot_index:
            return arr[pivot_index]
        elif k - 1 < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1

def run_performance_benchmarks():
    """
    Automated execution profile timing suite for large-scale data points.
    Isolates core loops cleanly to preserve timing metrics.
    """
    # -----------------------------------------------------------------
    # PART A: SEARCH ALGORITHM BENCHMARKS
    # -----------------------------------------------------------------
    search_sizes = [100, 1000, 10000, 1000000]
    benchmark_runs = 5
    print("\n" + "="*75)
    print("                 SCALABILITY RUNTIME BENCHMARK SUITE")
    print("="*75)
    print(f" {'Array Size':<12} | {'Scenario':<15} | {'Linear Search (ms)':<20} | {'Binary Search (ms)':<20} | {'Quickselect (ms)':<20}")
    print("-" * 75)
    
    for size in search_sizes:
        sorted_arr = list(range(size))
        unsorted_arr = sorted_arr.copy()
        random.shuffle(unsorted_arr)
        
        test_cases = {"Target Found": size // 2, "Target Missing": -99}
        
        for case, target in test_cases.items():
            linear_elapsed_ns = 0
            binary_elapsed_ns = 0
            quickselect_elapsed_ns = 0
            for _ in range(benchmark_runs):
                t0 = time.perf_counter_ns()
                for x in unsorted_arr:
                    if x == target:
                        break
                t1 = time.perf_counter_ns()
                linear_elapsed_ns += t1 - t0

                t2 = time.perf_counter_ns()
                low, high = 0, len(sorted_arr) - 1
                while low <= high:
                    mid = (low + high) // 2
                    if sorted_arr[mid] == target:
                        break
                    elif sorted_arr[mid] > target:
                        high = mid - 1
                    else:
                        low = mid + 1
                t3 = time.perf_counter_ns()
                binary_elapsed_ns += t3 - t2

                t4 = time.perf_counter_ns()
                try:
                    quickselect(unsorted_arr.copy(), size // 2)
                except ValueError:
                    pass
                t5 = time.perf_counter_ns()
                quickselect_elapsed_ns += t5 - t4

            lin_time_ms = linear_elapsed_ns / benchmark_runs / 1_000_000.0
            bin_time_ms = binary_elapsed_ns / benchmark_runs / 1_000_000.0
            quickselect_time_ms = quickselect_elapsed_ns / benchmark_runs / 1_000_000.0
            
            print(f" {size:<12} | {case:<15} | {lin_time_ms:<20.3f} | {bin_time_ms:<20.3f} | {quickselect_time_ms:<20.3f}")
            


def run_searching_demo():
    """
    Comprehensive Interactive User Interface Dashboard
    """
    while True:
        print("=========================================================")
        print("      SEARCHING INTERACTIVE ALGORITHM VISUALIZER    ")
        print("=========================================================")
        print("1] Run Visual Step-by-Step Search Demonstrations")
        print("2] Run Quickselect Demonstration")
        print("3] Run System Scalability Runtime Benchmarks (All Complexities)")
        print("0] Exit Searching Demo")
        print("=========================================================")
        
        choice = input("Select a dashboard menu option (1-5): ").strip()
        
        if choice == '0':
            print("Shutting down Interactive Module. Goodbye!")
            return
            
        elif choice == '1':
            print("\n--- Interactive Search Demonstration ---")
            demo_arr = [40, 10, 50, 20, 30, 60, 170, 80, 90, 100, 190, 120, 330, 140, 150]
            print(f"Base Unsorted Array: {demo_arr}")
            print(f"Base Sorted Array:   {sorted(demo_arr)}")

            try:
                target = int(input("Enter an integer value to find within these pools: "))
            except ValueError:
                print("Invalid integer entry.\n")
                continue
            linear_search_visual(demo_arr, target)
            binary_search_visual(sorted(demo_arr), target)

        elif choice == '2':
            print("\n--- Quickselect Demonstration ---")
            unsorted_sample = random.sample(range(1, 1000), 10)
            print(f"Generated Random Sample Array: {unsorted_sample}")
            try:
                k = int(input(f"Enter the (smallest) k-th position to find (1-{len(unsorted_sample)}): "))
                result = quickselect(unsorted_sample.copy(), k)
                print(f"The {k}-th smallest element in the array is: {result}\n")
            except ValueError as ve:
                print(f"Error: {ve}\n")

        elif choice == '3':
            run_performance_benchmarks()

        else:
            print("Invalid input selection. Please choose an integer between 0 and 3.\n")





# =====================================================
# HASH TABLE AND HEAP PRIORITY QUEUE IMPLEMENTATION
# =====================================================

Key = TypeVar("Key")
Value = TypeVar("Value")


class HashTable(Generic[Key, Value]):
    """A hash table that handles collisions with separate chaining."""

    def __init__(self, bucket_count: int = 17) -> None:
        if bucket_count <= 0:
            raise ValueError("bucket_count must be greater than zero")
        self._buckets: list[list[tuple[Key, Value]]] = [[] for _ in range(bucket_count)]
        self._size = 0

    def _bucket_index(self, key: Key) -> int:
        return hash(key) % len(self._buckets)

    def insert(self, key: Key, value: Value) -> None:
        """Add a pair, or update the value when the key already exists."""
        bucket = self._buckets[self._bucket_index(key)]
        for position, (stored_key, _) in enumerate(bucket):
            if stored_key == key:
                bucket[position] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1

    def search(self, key: Key) -> Value | None:
        """Return a value for a key, or None when it is not present."""
        for stored_key, value in self._buckets[self._bucket_index(key)]:
            if stored_key == key:
                return value
        return None

    def delete(self, key: Key) -> bool:
        """Remove a key and report whether it was found."""
        bucket = self._buckets[self._bucket_index(key)]
        for position, (stored_key, _) in enumerate(bucket):
            if stored_key == key:
                del bucket[position]
                self._size -= 1
                return True
        return False

    def items(self) -> Iterator[tuple[Key, Value]]:
        for bucket in self._buckets:
            yield from bucket

    @property
    def size(self) -> int:
        return self._size


@dataclass(order=True)
class PriorityItem(Generic[Value]):
    priority: int
    order: int
    value: Value


class PriorityQueue(Generic[Value]):
    """A min-priority queue implemented with a binary heap."""

    def __init__(self) -> None:
        self._heap: list[PriorityItem[Value]] = []
        self._next_order = 0

    def insert(self, value: Value, priority: int) -> None:
        self._heap.append(PriorityItem(priority, self._next_order, value))
        self._next_order += 1
        self._sift_up(len(self._heap) - 1)

    def _sift_up(self, child: int) -> None:
        while child > 0:
            parent = (child - 1) // 2
            if self._heap[parent] <= self._heap[child]:
                break
            self._heap[parent], self._heap[child] = self._heap[child], self._heap[parent]
            child = parent

    def _sift_down(self, parent: int) -> None:
        while True:
            smallest = parent
            left = parent * 2 + 1
            right = left + 1
            if left < len(self._heap) and self._heap[left] < self._heap[smallest]:
                smallest = left
            if right < len(self._heap) and self._heap[right] < self._heap[smallest]:
                smallest = right
            if smallest == parent:
                return
            self._heap[parent], self._heap[smallest] = self._heap[smallest], self._heap[parent]
            parent = smallest

    def delete(self) -> tuple[Value, int] | None:
        if not self._heap:
            return None
        first = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return first.value, first.priority

    def search(self, value: Value) -> tuple[Value, int] | None:
        for item in self._heap:
            if item.value == value:
                return item.value, item.priority
        return None

    def items(self) -> list[tuple[Value, int]]:
        return [(item.value, item.priority) for item in sorted(self._heap)]

    def __len__(self) -> int:
        return len(self._heap)


def build_demo_data(count: int = 100) -> list[tuple[str, str, int]]:
    """Generate a list of demo records with a predictable priority value."""
    return [
        (f"item-{number:03d}", f"Sample item {number}", (number % 10) + 1)
        for number in range(1, count + 1)
    ]


def benchmark_search(data: list[tuple[str, str, int]], repeats: int = 200) -> tuple[float, float]:
    """Compare hash lookup and linear search in milliseconds."""
    if not data:
        raise ValueError("data must contain at least one item")
    table = HashTable[str, str](bucket_count=17)
    records = [(key, description) for key, description, _ in data]
    for key, description in records:
        table.insert(key, description)
    target = records[-1][0]

    start = time.perf_counter()
    for _ in range(repeats):
        table.search(target)
    hash_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for _ in range(repeats):
        next((value for key, value in records if key == target), None)
    linear_ms = (time.perf_counter() - start) * 1000
    return hash_ms, linear_ms


def load_demo_data() -> tuple[HashTable[str, str], PriorityQueue[str]]:
    """Build a demo hash table and priority queue from sample data."""
    table: HashTable[str, str] = HashTable()
    queue: PriorityQueue[str] = PriorityQueue()
    for key, description, priority in build_demo_data():
        table.insert(key, description)
        queue.insert(key, priority)
    return table, queue


def print_contents(table: HashTable[str, str], queue: PriorityQueue[str]) -> None:
    """Display the current hash-table entries and queued tasks."""
    print(f"\nHash table ({table.size} items):")
    for key, value in sorted(table.items()):
        print(f"  {key}: {value}")
    print(f"\nPriority queue ({len(queue)} tasks, lowest number first):")
    for value, priority in queue.items():
        print(f"  Priority {priority}: {value}")



def run_hash_table_and_priority_queue_demo() -> None:
    """Demonstrate hash table and priority queue operations with timing."""
    table, queue = load_demo_data()
    print("Hash Tables and Priority Queues")
    print("100 sample items are loaded. Lower priority numbers are served first.")

    while True:
        print("\n1. Show items\n2. Add or update hash-table item\n3. Search hash table")
        print("4. Delete hash-table item\n5. Add priority-queue task\n6. Find priority-queue task")
        print("7. Serve next task\n8. Run search-speed comparison\n9. Load 100 sample items\n0. Exit")
        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            print_contents(table, queue)
        elif choice == "2":
            key = input("Key/name: ").strip()
            value = input("Value/description: ").strip()
            if key and value:
                table.insert(key, value)
                print(f"Stored '{key}'.")
            else:
                print("Both the key and value are required.")
        elif choice == "3":
            key = input("Key/name to find: ").strip()
            value = table.search(key)
            print(f"Found: {value}" if value is not None else "Item not found.")
        elif choice == "4":
            key = input("Key/name to delete: ").strip()
            print("Deleted." if table.delete(key) else "Item not found.")
        elif choice == "5":
            value = input("Task name: ").strip()
            try:
                priority = int(input("Priority (1 = most important): ").strip())
            except ValueError:
                print("Priority must be a whole number.")
                continue
            if value:
                queue.insert(value, priority)
                print(f"Added '{value}'.")
            else:
                print("A task name is required.")
        elif choice == "6":
            value = input("Task name to find: ").strip()
            result = queue.search(value)
            print(f"Found at priority {result[1]}." if result else "Task not found.")
        elif choice == "7":
            result = queue.delete()
            print(f"Served '{result[0]}' at priority {result[1]}." if result else "The queue is empty.")
        elif choice == "8":
            benchmark_data = build_demo_data()
            repeats = 200
            hash_ms, linear_ms = benchmark_search(benchmark_data, repeats=repeats)
            hash_per_search = hash_ms / repeats
            linear_per_search = linear_ms / repeats
            speed_ratio = linear_ms / hash_ms if hash_ms > 0 else float("inf")
            print("\nSearch performance comparison")
            print(f"  Dataset: {len(benchmark_data)} items")
            print(f"  Target: {benchmark_data[-1][0]} (last item, worst case for linear search)")
            print(f"  Repeated searches: {repeats}")
            print(f"  Hash-table total: {hash_ms:.3f} ms ({hash_per_search:.6f} ms/search)")
            print(f"  Linear-search total: {linear_ms:.3f} ms ({linear_per_search:.6f} ms/search)")
            if hash_ms < linear_ms:
                savings = (linear_ms - hash_ms) / linear_ms * 100 if linear_ms else 0
                print(f"  Result: hash-table search was {speed_ratio:.2f}x faster ({savings:.1f}% less time).")
            elif linear_ms < hash_ms:
                print(f"  Result: linear search was {hash_ms / linear_ms:.2f}x faster in this run.")
            else:
                print("  Result: both measurements were effectively equal in this run.")
            print("  Note: timings vary by computer; complexity predicts average O(1) hash lookup versus O(n) linear search.")
        elif choice == "9":
            table, queue = load_demo_data()
            print("Loaded 100 sample items into both structures.")
        elif choice == "0":
            print("Goodbye.")
            return
        else:
            print("Please choose a number from 0 to 9.")

# =========================================================
# END - HASH TABLE AND HEAP PRIORITY QUEUE IMPLEMENTATION
# =========================================================






# ==========================================
# SET - DATA STRUCTURE IMPLEMENTATION
# ==========================================


class CustomSet:
    def __init__(self, elements=None):
        """Initializes the set with an optional iterable of elements."""
        self._data = {}
        if elements:
            for element in elements:
                self.add(element)

    def add(self, element):
        """Adds an element to the set if it is not already present."""
        self._data[element] = True

    def remove(self, element):
        """Removes an element from the set. Raises KeyError if not found."""
        if element in self._data:
            del self._data[element]
        else:
            raise KeyError(f"Element {element} not found in set.")

    def discard(self, element):
        """Removes an element from the set if it is present, without error."""
        self._data.pop(element, None)

    def clear(self):
        """Remove all elements from the set in place."""
        self._data.clear()

    def contains(self, element):
        """Returns True if the element is in the set, False otherwise."""
        return element in self._data

    def size(self):
        """Returns the number of elements in the set."""
        return len(self._data)

    def to_list(self):
        """Returns a list of all elements in the set."""
        return list(self._data.keys())

    # --- Core Set Operations ---

    def union(self, other):
        """Returns a new set containing all unique elements from both sets."""
        new_set = CustomSet(self.to_list())
        for element in other.to_list():
            new_set.add(element)
        return new_set

    def intersection(self, other):
        """Returns a new set containing elements present in both sets."""
        new_set = CustomSet()
        # Iterate over the smaller set to optimize performance
        smaller, larger = (self, other) if self.size() < other.size() else (other, self)
        for element in smaller.to_list():
            if larger.contains(element):
                new_set.add(element)
        return new_set

    def difference(self, other):
        """Returns a new set with elements in this set but not in the other."""
        new_set = CustomSet()
        for element in self.to_list():
            if not other.contains(element):
                new_set.add(element)
        return new_set

    def symmetric_difference(self, other):
        """Returns elements in either set, but not in both."""
        # Concept: (A - B) union (B - A)
        diff_a = self.difference(other)
        diff_b = other.difference(self)
        return diff_a.union(diff_b)

    # --- Magic Methods for Pythonic Behavior ---

    def __str__(self):
        """String representation of the set."""
        if self.size() == 0:
            return "{}"
        return "{" + ", ".join(map(str, self._data.keys())) + "}"

    def __repr__(self):
        return f"CustomSet({self.to_list()})"

    def __contains__(self, element):
        """Supports Python's ``element in custom_set`` membership syntax."""
        return self.contains(element)

def run_set_demo():
    """
    Demonstrates the custom set implementation.
    Provides options to add, remove, and check membership of elements.
    """
    set_a = CustomSet([1, 2, 3, 4, 5])
    set_b = CustomSet([4, 5, 6, 7, 8])
    demo_set = CustomSet()
    print("\n--- Set Data Structure Demonstration ---")
    print("Set operations sample demonstration")
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Union of A and B: {set_a.union(set_b)}")
    print(f"Intersection of A and B: {set_a.intersection(set_b)}")
    print(f"Difference of A and B (A - B): {set_a.difference(set_b)}")
    print(f"Symmetric Difference of A and B: {set_a.symmetric_difference(set_b)}")
    print("\nYou can now interact with a custom set. You can add, remove, check membership, or clear the set.")
    while True:
        print("\nCurrent Set Contents:", demo_set)
        print("1] Add Element")
        print("2] Remove Element")
        print("3] Check Membership")
        print("4] Clear Set")
        print("0] Exit Set Demo")
        
        choice = input("Select an option (1-4, or 0 to exit): ").strip()
        
        if choice == '1':
            element = input("Enter element to add: ").strip()
            if element:
                demo_set.add(element)
                print(f"Added '{element}' to the set.")
            else:
                print("Element cannot be empty.")
            
        elif choice == '2':
            element = input("Enter element to remove: ").strip()
            if element in demo_set:
                demo_set.discard(element)
                print(f"Removed '{element}' from the set.")
            else:
                print(f"Element '{element}' not found in the set.")
                
        elif choice == '3':
            element = input("Enter element to check: ").strip()
            if element in demo_set:
                print(f"'{element}' is in the set.")
            else:
                print(f"'{element}' is not in the set.")
                
        elif choice == '4':
            demo_set.clear()
            print("Set has been cleared.")
            
        elif choice == '0':
            print("Exiting Set Demo. Goodbye!")
            return
            
        else:
            print("Invalid option. Please choose 1-4, or 0 to exit.")
    


if __name__ == "__main__":
    while True:
        print("\n===========================================================================================")
        print("    DATA STRUCTURE SCALED TIME & SPACE BENCHMARK TOOL and ALGORITHM PERFORMANCE VISUALIZER")
        print("===========================================================================================")
        print("1. Linear Data Structures (Arrays, Stacks, Queues, Linked Lists)")
        print("2. Non-Linear Data Structures (Binary Trees, Graphs)")
        print("3. Sorting Algorithms Demo and Performance Benchmark")
        print("4. Searching Algorithms Demo and Performance Benchmark")
        print("5. Hash Tables and Priority Queues Demo")
        print("6. Set Data Structure Demo")
        print("0. Exit Application")
        main_choice = input("Choose a data structure category to analyze: ").strip()

        if main_choice == '1':
            linear_data_structure_implementation()
        elif main_choice == '2':
            non_linear_data_structure_implementation()
        elif main_choice == '3':
            run_sorting_demo()
        elif main_choice == '4':
            run_searching_demo()
        elif main_choice == '5':
            run_hash_table_and_priority_queue_demo()
        elif main_choice == '6':
            run_set_demo()
        elif main_choice == '0':
            print("\nLeaving Data Structures and Algorithms Demo. Happy engineering!")
            print("Exiting Application. Goodbye!")
            sys.exit()
        else:
            print("Invalid option. Please choose 1-7 or 0 to exit application.")
