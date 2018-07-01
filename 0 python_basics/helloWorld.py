import copy

def quicksort(arr, lo, hi):
    if lo == hi:
        return
    else:
        swap(arr[lo:hi + 1])


def swap(arr):
    # set pivot to the last element of the array
    small_vals = []
    large_vals = []
    pivot_val  = arr[-1]
    pivot_indx = len(arr) - 1
    for i in range(len(arr) - 1):
        if arr[i] > pivot_val:
            pivot_indx -= 1
            large_vals.append(arr[i])
        else:
            small_vals.append(arr[i])

    arr[:len(small_vals)]     = small_vals
    arr[len(small_vals)]      = pivot_val
    arr[len(small_vals) + 1:] = large_vals
    quicksort(arr,0,len(small_vals) - 1)
    quicksort(arr,len(small_vals) + 1, len(arr) - 1)


def bubblesort(arr):
    print("Original array: {}".format(arr))
    sorted_arr = arr
    for i in range(len(arr)):
        sorted_arr[i] = min(arr[i:])

    print("Sorted array: {}".format(sorted_arr))

def dumbsort(arr):
    print("Original array: {}".format(arr))
    sorted_arr   = []
    copy_arr     = arr
    num_elements = len(arr)

    for i in range(num_elements):
        indx          = find_index_of_min(copy_arr)
        sorted_arr.append(copy_arr.pop(indx))

    print("Sorted array: {}".format(sorted_arr))

def find_index_of_min(arr):
    min      = arr[0]
    min_indx = 0
    for i in range(len(arr)):
        if min > arr[i]:
            min = arr[i]
            min_indx = i
    
    return min_indx


#=========================================
# Test area
#=========================================
test_arr   = [4,1,6,7,2,0,3]
sorted_arr = [0,1,2,4,6,7]

#print("Dumbsort")
#dumbsort(copy.deepcopy(test_arr))
#print("Bubblesort")
#bubblesort(copy.deepcopy(test_arr))
quicksort(test_arr, 0 , len(test_arr) - 1)
print(test_arr)