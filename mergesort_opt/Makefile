CC 	 = gcc
CLCC	 = clcc
CFLAGS 	 = -lm -lOpenCL -O3 
INCLUDE  = -I/usr/local/cuda/include

all: mergesort mergesort.ptx

mergesort: mergesort.c 
	$(CC) -o $@ $< $(LIB_PATH) $(CFLAGS) $(INCLUDE)

mergesort.ptx: mergesort.cl
	$(CLCC) -o $@ -c $<

clean:
	rm -rf mergesort mergesort.ptx
