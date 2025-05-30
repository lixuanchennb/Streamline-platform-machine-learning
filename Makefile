.PHONY: all clean

all:
	gcc -o main main.c

clean:
	rm -f main