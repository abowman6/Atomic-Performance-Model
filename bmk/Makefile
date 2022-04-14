
all: atomic_bmk atomic_S atomic_random

FLAGS= -Xptxas -O3

atomic_bmk: atomic.cu
	nvcc $(FLAGS) -o $@ $<

atomic_random: atomic_random.cu
	nvcc $(FLAGS) -o $@ $<

atomic_S: atomic_S.cu
	nvcc $(FLAGS) -o $@ $<

clean:
	rm -f atomic_bmk atomic_S atomic_random




