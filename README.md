# CUDA_Data_Parallel_Reduction

**About:**

Implementation of a work-efficient parallel reduction algorithm on the GPU with accumulation using atomic additions. An alternative version would invoke the reduction kernal more than once in a hierarchical manner so as to further reduce the block sums computed by the previous kernel invocations.

**Execution:**

* Run "make" to build the executable of this file.
* For debugging, run "make dbg=1" to build a debuggable version of the executable binary.
* Run the binday using the command "./vector_reduction"

There are several modes of operation for the application:
* No arguments: The application will create two randomly sized and ini- tialized matrices such that the matrix operation M * N is valid, and P is properly sized to hold the result. After the device multiplication is in- voked, it will compute the correct solution matrix using the CPU, and compare that solution with the device-computed solution. If it matches (within a certain tolerance), it will print out “Test PASSED” to the screen before exiting.
* One argument: The application will initialize the input array with the values found in the file specified by the argument.

In either mode, the program will print out the final result of the CPU and GPU computations, and whether or not the comparison passed.
