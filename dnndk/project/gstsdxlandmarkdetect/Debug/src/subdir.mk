################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/densebox.cpp \
../src/gstsdxlandmarkdetect.cpp 

OBJS += \
./src/densebox.o \
./src/gstsdxlandmarkdetect.o 

CPP_DEPS += \
./src/densebox.d \
./src/gstsdxlandmarkdetect.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: ARM A53 Linux g++ compiler'
	aarch64-linux-gnu-g++ -Wall -O2 -I"/home/dehim/Software/DNNDK2/include" -I/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux/usr/include -I/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux/usr/include/glib-2.0 -I/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux/usr/lib/glib-2.0/include -I/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux/usr/include/gstreamer-1.0 -I/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux/usr/lib/gstreamer-1.0/include -c -fPIC -fpermissive -fmessage-length=0 -MT"$@" --sysroot=/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


