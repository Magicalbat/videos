#version 440 

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (std430, binding = 0) readonly buffer InputVecA {
    float a_data[];
};

layout (std430, binding = 1) readonly buffer InputVecB {
    float b_data[];
};

layout (std430, binding = 2) writeonly buffer OutputVec {
    float out_data[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;

    out_data[index] = a_data[index] + b_data[index];
}

