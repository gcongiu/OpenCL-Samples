TARGET	 = clcc
SRC	 = clcc.c
CC	 = gcc
CFLAGS	 = -fPIC -lOpenCL 
INC_PATH = -I/usr/local/cuda/include 

ifeq ($(dbg),1)
	OPTIONS = -g -D_ENABLE_DEBUG_
endif

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(CFLAGS) $(INC_PATH) $(OPTIONS) 

clean:
	rm -f $(TARGET)
