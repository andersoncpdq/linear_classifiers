################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../classifiers/perceptron/perceptron.cpp 

OBJS += \
./classifiers/perceptron/perceptron.o 

CPP_DEPS += \
./classifiers/perceptron/perceptron.d 


# Each subdirectory must supply rules for building sources it contributes
classifiers/perceptron/%.o: ../classifiers/perceptron/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


