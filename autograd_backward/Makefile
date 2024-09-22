TARGET := main
OBJECTS := main.cpp

LIBS := -lm -ldl

CPPFLAGS := -Wall -Werror -std=gnu++2b -O2 -Wextra

CPP := g++

all: $(OBJECTS)
	$(CPP) $(CPPFLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

.PHONY: clean help

clean:
	rm -f $(TARGET)