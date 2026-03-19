
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef i8 b8;
typedef i32 b32;

typedef float f32;

#define UNICODE
#include <Windows.h>
#include <gl/GL.h>

#include "opengl_api.h"

#define WGL_CONTEXT_MAJOR_VERSION_ARB           0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB           0x2092
typedef HGLRC (wglCreateContextAttribsARB_func)(HDC hDC, HGLRC hshareContext, const int *attribList);

#define WGL_DRAW_TO_WINDOW_ARB                  0x2001
#define WGL_SUPPORT_OPENGL_ARB                  0x2010
#define WGL_DOUBLE_BUFFER_ARB                   0x2011
#define WGL_COLOR_BITS_ARB                      0x2014
#define WGL_DEPTH_BITS_ARB                      0x2022
#define WGL_STENCIL_BITS_ARB                    0x2023
#define WGL_PIXEL_TYPE_ARB                      0x2013
#define WGL_TYPE_RGBA_ARB                       0x202B

typedef BOOL (wglChoosePixelFormatARB_func)(HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);

static wglCreateContextAttribsARB_func* wglCreateContextAttribsARB = NULL;
static wglChoosePixelFormatARB_func* wglChoosePixelFormatARB = NULL;

#define DUMMY_CLASS_NAME L"dummy_wnd_class"
#define REAL_CLASS_NAME L"real_wnd_class"

static LRESULT window_proc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

i32 pixel_format = 0;
b32 should_close = false;

const char* vert_shader_source = "#version 330 core\n"
    "layout (location = 0) in vec2 a_pos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(a_pos.x, a_pos.y, 0.0, 1.0);\n"
    "}\0";
const char* frag_shader_source = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

int main(void) {
    HINSTANCE module_handle = GetModuleHandleW(NULL);

    SetProcessDpiAwarenessContext(
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    );

    // Dummy window -> load extensions
    {
        WNDCLASSW dummy_wnd_class = {
            .lpszClassName = DUMMY_CLASS_NAME,
            .hInstance = module_handle,
            .lpfnWndProc = DefWindowProcW
        };
        RegisterClassW(&dummy_wnd_class);

        HWND dummy_wnd = CreateWindowW(
            DUMMY_CLASS_NAME, L"", WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
            NULL, NULL, NULL, NULL
        );

        HDC dummy_dc = GetDC(dummy_wnd);

        PIXELFORMATDESCRIPTOR pfd = {
            .nSize = sizeof(PIXELFORMATDESCRIPTOR),
            .nVersion = 1,
            .dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
            .iPixelType = PFD_TYPE_RGBA,
            .cColorBits = 32
        };
        i32 pf = ChoosePixelFormat(dummy_dc, &pfd);
        DescribePixelFormat(dummy_dc, pf, sizeof(PIXELFORMATDESCRIPTOR), &pfd);
        SetPixelFormat(dummy_dc, pf, &pfd);

        HGLRC dummy_gl_context = wglCreateContext(dummy_dc);
        wglMakeCurrent(dummy_dc, dummy_gl_context);

        wglCreateContextAttribsARB = (wglCreateContextAttribsARB_func*)wglGetProcAddress("wglCreateContextAttribsARB");
        wglChoosePixelFormatARB = (wglChoosePixelFormatARB_func*)wglGetProcAddress("wglChoosePixelFormatARB");

        i32 pf_attribs[] = {
            WGL_DRAW_TO_WINDOW_ARB, GL_TRUE, 
            WGL_SUPPORT_OPENGL_ARB, GL_TRUE, 
            WGL_DOUBLE_BUFFER_ARB, GL_TRUE, 
            WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
            WGL_COLOR_BITS_ARB, 32,
            WGL_DEPTH_BITS_ARB, 24,
            WGL_STENCIL_BITS_ARB, 8,
            0
        };

        u32 num_formats = 0;
        wglChoosePixelFormatARB(dummy_dc, pf_attribs, NULL, 1, &pixel_format, &num_formats);

        // Load in OpenGL functions
        #define X(ret, name, args) name = (gl_##name##_func*)wglGetProcAddress(#name);
        #include "opengl_funcs_xlist.h"
        #undef X

        wglMakeCurrent(dummy_dc, NULL);
        wglDeleteContext(dummy_gl_context);
        ReleaseDC(dummy_wnd, dummy_dc);
        DestroyWindow(dummy_wnd);
        UnregisterClassW(DUMMY_CLASS_NAME, module_handle);
    }

    WNDCLASSW wnd_class = {
        .lpszClassName = REAL_CLASS_NAME,
        .hInstance = module_handle,
        .lpfnWndProc = window_proc,
        .hCursor = LoadCursorW(0, IDC_ARROW),
    };
    RegisterClassW(&wnd_class);

    HWND window = CreateWindowW(
        REAL_CLASS_NAME, L"WGL Example", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 720,
        NULL, NULL, NULL, NULL
    );

    HDC dc = GetDC(window);
    
    PIXELFORMATDESCRIPTOR pfd = { 0 };
    DescribePixelFormat(dc, pixel_format, sizeof(PIXELFORMATDESCRIPTOR), &pfd);
    SetPixelFormat(dc, pixel_format, &pfd);
    
    i32 context_attribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 5,
        0
    };

    HGLRC gl_context = wglCreateContextAttribsARB(dc, NULL, context_attribs);
    wglMakeCurrent(dc, gl_context);

    ShowWindow(window, SW_SHOW);

    u32 shader_program = 0;
    {
        i32 success = 0;
        char info_log[512];

        u32 vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vert_shader_source, NULL);
        glCompileShader(vertex_shader);
        glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertex_shader, 512, NULL, info_log);
            printf("Vertex shader failed to compile: %s\n", info_log);
        }
        // fragment shader
        unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &frag_shader_source, NULL);
        glCompileShader(fragment_shader);
        // check for shader compile errors
        glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragment_shader, 512, NULL, info_log);
            printf("Fragment shader failed to compile: %s\n", info_log);
        }
        // link shaders
        shader_program = glCreateProgram();
        glAttachShader(shader_program, vertex_shader);
        glAttachShader(shader_program, fragment_shader);
        glLinkProgram(shader_program);
        // check for linking errors
        glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader_program, 512, NULL, info_log);
            printf("Shader program failed to link: %s\n", info_log);
        }
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
    }

    u32 VBO = 0, VAO = 0;
    {
        f32 vertices[] = {
            -0.5f, -0.5f,
             0.0f,  0.5f,
             0.5f, -0.5f,
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0); 
    }

    glClearColor(0, 0.2, 0.2, 1.0);
    while (!should_close) {
        MSG msg = { 0 };
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        // Drawing
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shader_program);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        SwapBuffers(dc);

        Sleep(15);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shader_program);


    wglMakeCurrent(dc, NULL);
    wglDeleteContext(gl_context);
    ReleaseDC(window, dc);
    DestroyWindow(window);
    UnregisterClassW(REAL_CLASS_NAME, module_handle);

    return 0;
}

static LRESULT window_proc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE: {
            should_close = true;
        } break;
    }

    return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

