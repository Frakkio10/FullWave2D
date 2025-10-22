# compile C files located in src by doing `make`

# Define the directory to the src/ folder
SRCDIR = fullwave2d/src

compile: $(SRCDIR)/maxwell_2d_omode.c $(SRCDIR)/maxwell_2d_xmode.c $(SRCDIR)/py_maxwell_interface.c
	gcc -Wall -w -c -fPIC $(SRCDIR)/maxwell_2d_omode.c $(SRCDIR)/maxwell_2d_xmode.c $(SRCDIR)/py_maxwell_interface.c
	gcc -Wall -shared -o $(SRCDIR)/py_maxwell_interface.so maxwell_2d_omode.o maxwell_2d_xmode.o py_maxwell_interface.o -lm
	make clean

compileO: $(SRCDIR)/maxwell_2d_omode.c $(SRCDIR)/py_maxwell_interface.c
	gcc -Wall -w -c -fPIC $(SRCDIR)/maxwell_2d_omode.c $(SRCDIR)/py_maxwell_interface.c
	gcc -Wall -shared -o $(SRCDIR)/py_maxwell_interface.so maxwell_2d_omode.o py_maxwell_interface.o -lm
	make clean

compileX: $(SRCDIR)/maxwell_2d_xmode.c $(SRCDIR)/py_maxwell_interface.c
	gcc -Wall -w -c -fPIC $(SRCDIR)/maxwell_2d_xmode.c $(SRCDIR)/py_maxwell_interface.c
	gcc -Wall -shared -o $(SRCDIR)/py_maxwell_interface.so maxwell_2d_xmode.o py_maxwell_interface.o -lm
	make clean

clean-pycache:
	find . -type d -name '__pycache__' -exec rm -r {} +

clean:
	rm *.o