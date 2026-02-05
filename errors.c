#include <stdio.h>
#include <stdio.h>
#include <stdint.h>

#include <Windows.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;

typedef struct {
    u8* str;
    u64 size;
} string8;

#define STR8_LIT(s) (string8){ (u8*)(s), sizeof(s) - 1 }
#define STR8_FMT(s) (int)(s).size, (char*)(s).str

typedef enum {
    LOG_LEVEL_NONE = 0,

    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR
} log_level;

typedef struct {
    string8 msg;
    log_level level;
} log_msg;

typedef struct {
    u32 size;
    u32 capacity;
    log_msg* msgs;
} log_frame;

typedef struct {
    u32 size;
    u32 capacity;
    log_frame* frames;
} log_context;

void log_frame_begin(void);
string8 log_frame_peek(u32 level_mask);
string8 log_frame_end(u32 level_mask);

#define error_emit(msg) log_emit(LOG_LEVEL_ERROR, (msg))

void log_emit(log_level level, string8 msg);

string8 file_read(const char* file_name);

int main(int argc, char** argv) {
    if (argc <= 1) { return 0; }

    log_frame_begin();

    string8 file = file_read(argv[1]);

    u32 file_sum = 0;
    for (u64 i = 0; i < file.size; i++) {
        file_sum += file.str[i];
    }

    printf("File sum: %u\n", file_sum);

    if (file.size) {
        free(file.str);
    }

    string8 logs = log_frame_end(LOG_LEVEL_ERROR);
    if (logs.size) {
        printf("Error: %.*s\n", STR8_FMT(logs));
        free(logs.str);
    }

    return 0;
}

string8 file_read(const char* file_name) {
    HANDLE file_handle = CreateFileA(
        file_name, GENERIC_READ, 0, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );

    if (file_handle == INVALID_HANDLE_VALUE) {
        error_emit(STR8_LIT("Failed to open file"));
        return (string8){ 0 };
    }

    u32 file_size = 0;
    if ((file_size = GetFileSize(file_handle, NULL)) == INVALID_FILE_SIZE) {
        error_emit(STR8_LIT("Failed to get file size"));
        CloseHandle(file_handle);
        return (string8){ 0 };
    }

    string8 out = {
        .size = file_size,
        .str = (u8*)malloc(file_size)
    };

    if (!ReadFile(file_handle, out.str, file_size, NULL, NULL)) {
        error_emit(STR8_LIT("Failed to read data from file"));
        free(out.str);
        out = (string8){ 0 };
    }

    CloseHandle(file_handle);
    return out;
}

static __thread log_context _log_context = { 0 };

log_frame _log_create_frame(void) {
    log_frame frame = {
        .size = 0,
        .capacity = 4,
    };

    frame.msgs = (log_msg*)malloc(sizeof(log_msg) * frame.capacity);

    return frame;
}

void log_frame_begin(void) {
    if (_log_context.frames == NULL) {
        _log_context.capacity = 4;
        _log_context.size = 0;
        _log_context.frames = (log_frame*)malloc(
            sizeof(log_frame) * _log_context.capacity
        );
    }

    if (_log_context.size >= _log_context.capacity) {
        _log_context.capacity *= 2;
        _log_context.frames = (log_frame*)realloc(
            _log_context.frames,
            sizeof(log_frame) * _log_context.capacity
        );
    }

    _log_context.frames[_log_context.size++] = _log_create_frame();
}

string8 log_frame_peek(u32 level_mask) {
    if (_log_context.frames == NULL) { return (string8){ 0 }; }

    log_frame* frame = &_log_context.frames[_log_context.size-1];

    u32 num_logs_in_mask = 0;
    u64 total_out_size = 0;

    for (u32 i = 0; i < frame->size; i++) {
        if ((frame->msgs[i].level & level_mask) == 0) {
            continue;
        }

        num_logs_in_mask++;
        total_out_size += frame->msgs[i].msg.size;
    }

    if (num_logs_in_mask == 0) {
        return (string8){ 0 };
    }

    total_out_size += num_logs_in_mask - 1;

    string8 out = {
        .size = total_out_size,
        .str = (u8*)malloc(total_out_size)
    };

    u64 out_pos = 0;

    for (u32 i = 0; i < frame->size; i++) {
        log_msg* msg = &frame->msgs[i];

        if ((msg->level & level_mask) == 0) {
            continue;
        }

        if (out_pos != 0) {
            out.str[out_pos++] = '\n';
        }

        memcpy(out.str + out_pos, msg->msg.str, msg->msg.size);
        out_pos += msg->msg.size;
    }

    return out;
}

string8 log_frame_end(u32 level_mask) {
    if (_log_context.frames == NULL) { return (string8){ 0 }; }

    string8 out = log_frame_peek(level_mask);

    free(_log_context.frames[_log_context.size-1].msgs);
    _log_context.size--;

    return out;
}

void log_emit(log_level level, string8 msg) {
    if (_log_context.frames == NULL) { return; }

    log_frame* frame = &_log_context.frames[_log_context.size-1];

    if (frame->size >= frame->capacity) {
        frame->capacity *= 2;
        frame->msgs = (log_msg*)realloc(
            frame->msgs, sizeof(log_msg) * frame->capacity
        );
    }

    frame->msgs[frame->size++] = (log_msg){
        .level = level,
        .msg = msg
    };
}

