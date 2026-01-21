#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include <vulkan/vulkan.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t b32;

typedef float f32;

#define ALIGN_UP(n, a) (((n) + (a) - 1) - ((n) + (a) - 1) % (a))

int main(void) {
    const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
    b32 use_validation_layer = false;

    VkInstance instance = NULL;
    {
        VkInstanceCreateInfo instance_create_info = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &(VkApplicationInfo){
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .apiVersion = VK_API_VERSION_1_0
            }
        };

        #ifndef NDEBUG


        u32 num_layers = 0;
        vkEnumerateInstanceLayerProperties(&num_layers, NULL);

        VkLayerProperties* layer_props = (VkLayerProperties*)malloc(sizeof(VkLayerProperties) * num_layers);
        vkEnumerateInstanceLayerProperties(&num_layers, layer_props);

        for (u32 i = 0; i < num_layers; i++) {
            if (strcmp(layer_props[i].layerName, validation_layer_name) == 0) {
                use_validation_layer = true;
                break;
            }
        }

        free(layer_props);

        if (use_validation_layer) {
            instance_create_info.enabledLayerCount = 1;
            instance_create_info.ppEnabledLayerNames = &validation_layer_name;
        }

        #endif

        vkCreateInstance(&instance_create_info, NULL, &instance);
    }

    VkPhysicalDevice physical_device = NULL;
    {
        u32 num_physical_devices = 1;
        vkEnumeratePhysicalDevices(instance, &num_physical_devices, &physical_device);

        VkPhysicalDeviceProperties physical_device_properties = { 0 };
        vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

        printf("Using physical device: %s\n", physical_device_properties.deviceName);
    }

    u32 queue_family_index = 0;
    VkDevice device = NULL;
    {
        u32 queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, NULL);

        VkQueueFamilyProperties* queue_props = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_props);

        for (u32 i = 0; i < queue_family_count; i++) {
            if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family_index = i;
                break;
            }
        }

        free(queue_props);

        f32 priority = 0.0f;

        VkDeviceCreateInfo device_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queue_family_index,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            },
        };

        #ifndef NDEBUG
        if (use_validation_layer) {
            device_create_info.enabledLayerCount = 1;
            device_create_info.ppEnabledLayerNames = &validation_layer_name;
        }
        #endif

        vkCreateDevice(physical_device, &device_create_info, NULL, &device);
    }

    u32 vector_size = 16;
    u32 buffer_size = sizeof(f32) * vector_size;

    VkBuffer in_buffer = NULL, out_buffer = NULL;
    {
        VkBufferCreateInfo buffer_create_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = buffer_size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_family_index
        };

        vkCreateBuffer(device, &buffer_create_info, NULL, &in_buffer);
        vkCreateBuffer(device, &buffer_create_info, NULL, &out_buffer);
    }

    VkMemoryRequirements in_mem_reqs = { 0 }, out_mem_reqs = { 0 };
    vkGetBufferMemoryRequirements(device, in_buffer, &in_mem_reqs);
    vkGetBufferMemoryRequirements(device, out_buffer, &out_mem_reqs);

    u64 in_buffer_offset = 0;
    u64 in_buffer_size = in_mem_reqs.size;
    u64 out_buffer_offset = ALIGN_UP(in_mem_reqs.size, out_mem_reqs.alignment);
    u64 out_buffer_size = out_mem_reqs.size;
    u64 total_memory_size = out_buffer_offset + out_buffer_size;

    VkPhysicalDeviceMemoryProperties mem_props = { 0 };
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    u32 mem_type_index = 0;
    for (u32 i = 0; i < mem_props.memoryTypeCount; i++) {
        if (
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && 
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        ) {
            mem_type_index = i;
            break;
        }
    }

    VkDeviceMemory memory = NULL;
    vkAllocateMemory(
        device, &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .memoryTypeIndex = mem_type_index,
            .allocationSize = total_memory_size
        }, NULL, &memory
    );

    {
        f32* data = NULL;
        vkMapMemory(device, memory, 0, total_memory_size, 0, (void**)(&data));

        f32* in_data = data;
        f32* out_data = data + (out_buffer_offset / sizeof(f32));

        for (u32 i = 0; i < vector_size; i++) {
            in_data[i] = i;
            out_data[i] = vector_size - i - 1;
        }

        vkUnmapMemory(device, memory);
    }

    vkBindBufferMemory(device, in_buffer, memory, 0);
    vkBindBufferMemory(device, out_buffer, memory, out_buffer_offset);

    printf("Initial Memory\n");
    {
        f32* data = NULL;
        vkMapMemory(device, memory, 0, total_memory_size, 0, (void**)(&data));

        f32* in_data = data;
        f32* out_data = data + (out_buffer_offset / sizeof(f32));

        printf("In Data : [ "); 
        for (u32 i = 0; i < vector_size; i++) {
            printf("%2.0f ", in_data[i]);
        }
        printf("]\n");

        printf("Out Data: [ "); 
        for (u32 i = 0; i < vector_size; i++) {
            printf("%2.0f ", out_data[i]);
        }
        printf("]\n");


        vkUnmapMemory(device, memory);
    }

    VkShaderModule shader_module = NULL;
    {
        FILE* f = fopen("add.spv", "rb");

        fseek(f, 0, SEEK_END);
        u64 size = ftell(f);
        fseek(f, 0, SEEK_SET);

        u8* shader_file = (u8*)malloc(size);
        fread(shader_file, 1, size, f);

        vkCreateShaderModule(
            device, &(VkShaderModuleCreateInfo){
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size,
                .pCode = (u32*)shader_file,
            }, NULL, &shader_module
        );

        free(shader_file);
    }

    VkDescriptorSetLayoutBinding bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };

    VkDescriptorSetLayout descriptor_set_layout = NULL;
    vkCreateDescriptorSetLayout(
        device, &(VkDescriptorSetLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
            .pBindings = bindings,
        }, NULL, &descriptor_set_layout
    );

    VkDescriptorPool descriptor_pool = NULL;
    vkCreateDescriptorPool(
        device, &(VkDescriptorPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &(VkDescriptorPoolSize){
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = sizeof(bindings) / sizeof(bindings[0])
            }
        }, NULL, &descriptor_pool
    );

    VkDescriptorSet descriptor_set = NULL;
    vkAllocateDescriptorSets(
        device, &(VkDescriptorSetAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout
        }, &descriptor_set
    );

    VkWriteDescriptorSet write_sets[] = {
        (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &(VkDescriptorBufferInfo){
                .buffer = in_buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            }
        },
        (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &(VkDescriptorBufferInfo){
                .buffer = out_buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            }
        },
        (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &(VkDescriptorBufferInfo){
                .buffer = out_buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            }
        },
    };

    vkUpdateDescriptorSets(
        device, sizeof(write_sets) / sizeof(write_sets[0]),
        write_sets, 0, NULL
    );

    VkPipelineLayout pipeline_layout = NULL;
    vkCreatePipelineLayout(
        device, &(VkPipelineLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        }, NULL, &pipeline_layout
    );

    VkPipeline pipeline = NULL;
    vkCreateComputePipelines(
        device, NULL, 1, &(VkComputePipelineCreateInfo){
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .layout = pipeline_layout,
            .stage = (VkPipelineShaderStageCreateInfo){
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main"
            },
        }, NULL, &pipeline
    );

    VkCommandPool cmd_pool = NULL;
    vkCreateCommandPool(
        device, &(VkCommandPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = queue_family_index
        }, NULL, &cmd_pool
    );

    VkCommandBuffer cmd_buffer = NULL;
    vkAllocateCommandBuffers(
        device, &(VkCommandBufferAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cmd_pool,
            .commandBufferCount = 1,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY
        }, &cmd_buffer
    );

    vkBeginCommandBuffer(
        cmd_buffer, &(VkCommandBufferBeginInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        }
    );

    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(
        cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout, 0, 1, &descriptor_set,
        0, NULL
    );
    vkCmdDispatch(cmd_buffer, vector_size, 1, 1);

    vkEndCommandBuffer(cmd_buffer);

    VkQueue queue = NULL;
    vkGetDeviceQueue(device, queue_family_index, 0, &queue);

    VkFence fence = NULL;
    vkCreateFence(
        device, &(VkFenceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        }, NULL, &fence
    );

    vkQueueSubmit(
        queue, 1, &(VkSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd_buffer
        }, fence
    );
    vkWaitForFences(device, 1, &fence, true, ~(u64)(0));

    printf("Final Memory\n");
    {
        f32* data = NULL;
        vkMapMemory(device, memory, 0, total_memory_size, 0, (void**)(&data));

        f32* in_data = data;
        f32* out_data = data + (out_buffer_offset / sizeof(f32));

        printf("In Data : [ "); 
        for (u32 i = 0; i < vector_size; i++) {
            printf("%2.0f ", in_data[i]);
        }
        printf("]\n");

        printf("Out Data: [ "); 
        for (u32 i = 0; i < vector_size; i++) {
            printf("%2.0f ", out_data[i]);
        }
        printf("]\n");


        vkUnmapMemory(device, memory);
    }

    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmd_pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyDescriptorPool(device, descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, NULL);
    vkDestroyShaderModule(device, shader_module, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyBuffer(device, in_buffer, NULL);
    vkDestroyBuffer(device, out_buffer, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    return 0;
}

