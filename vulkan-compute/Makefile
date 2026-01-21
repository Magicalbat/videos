
VULKAN_SDK_PATH = C:\VulkanSDK\1.4.335.0

CC = clang
CFLAGS = -m64 -std=c99 -Wall -Wextra -pedantic -I$(VULKAN_SDK_PATH)/Include
DEBUG_FLAGS = -DDEBUG -O0 -g -fsanitize=address
RELEASE_FLAGS = -DNDEBUG

LFLAGS = -L$(VULKAN_SDK_PATH)/Lib -lvulkan-1.lib

config ?= debug

ifeq ($(config), debug)
	CFLAGS += $(DEBUG_FLAGS)
else
	CFLAGS += $(RELEASE_FLAGS)
endif

all: vulkan-compute

vulkan-compute:
	$(CC) main.c $(CFLAGS) $(LFLAGS) -o vulkan-compute.exe

.PHONY: all vulkan-compute

