################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Main.cpp \
../Matrix.cpp \
../NeuralNetwork.cpp \
../Neuron.cpp \
../SigmoidLayer.cpp \
../Util.cpp \
../WeightsAndBiasUpdates.cpp 

OBJS += \
./Main.o \
./Matrix.o \
./NeuralNetwork.o \
./Neuron.o \
./SigmoidLayer.o \
./Util.o \
./WeightsAndBiasUpdates.o 

CPP_DEPS += \
./Main.d \
./Matrix.d \
./NeuralNetwork.d \
./Neuron.d \
./SigmoidLayer.d \
./Util.d \
./WeightsAndBiasUpdates.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


