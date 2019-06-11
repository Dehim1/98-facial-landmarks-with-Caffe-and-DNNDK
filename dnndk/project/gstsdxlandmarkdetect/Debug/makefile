################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: libgstsdxlandmarkdetect.so

# Tool invocations
libgstsdxlandmarkdetect.so: $(OBJS)  $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: ARM A53 Linux g++ linker'
	aarch64-linux-gnu-g++ -L"/home/dehim/Software/DNNDK2/lib" -shared --sysroot=/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux -L=/lib -L=/usr/lib -Wl,-rpath-link=/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux/lib,-rpath-link=/home/dehim/install/Xilinx/SDx/2018.3/platforms/zcu102-rv-ss-2018-3-dnndk/zcu102_rv_ss/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux/usr/lib -o "libgstsdxlandmarkdetect.so" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(EXECUTABLES)$(OBJS)$(C_UPPER_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C_DEPS)$(CPP_DEPS) libgstsdxlandmarkdetect.so
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
