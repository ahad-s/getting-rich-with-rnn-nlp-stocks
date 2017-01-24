CXX = g++-5
CXXFLAGS = -std=c++14 -MMD
EXEC = skynet
OBJECTS = skynetPREDICT.o RNN.o
DEPENDS = ${OBJECT:.o=.d}

$EXEC: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

-include ${DEPENDS}

.PHONY: clean

clean:
	rm ${OBJECTS} ${EXEC} ${DEPENDS}
