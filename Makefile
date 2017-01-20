CXX = g++-5
CXXFLAGS = -std=c++14 -Wall -MMD
EXEC = skynet
OBJECTS = skynetPREDICT.o
DEPENDS = ${OBJCET:.o=.d}

$EXEC: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

-include ${DEPENDS}

.PHONY: clean

clean:
	rm ${OBJECTS} ${EXEC} ${DEPENDS}