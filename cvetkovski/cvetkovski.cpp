#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <numeric>
#include <vector>

#include "pathconfig.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "indexbuffer.h"
#include "renderer.h"
#include "shader.h"
#include "vertexarray.h"
#include "vertexbuffer.h"
#include "vertexbufferlayout.h"

#include "sphere.h"
#include "texture.h"

const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 768;

std::vector<float> linspace(float start_in, float end_in, size_t num_in)
{
    const float dx = (end_in - start_in) / (num_in - 1);
    std::vector<float> linspaced(num_in);
    int iter = 0;
    std::generate(linspaced.begin(), linspaced.end(), [&] { return start_in + (iter++) * dx; });
    return linspaced;
}

float binomial(long n, long i)
{
    long brojnik = 1;
    for (long f = n; f >= n - i + 1; f--)
    {
        brojnik *= f;
    }
    long long nazivnik = 1;
    for (long f = i; f >= 1; f--)
    {
        nazivnik *= f;
    }
    return float(brojnik) / float(nazivnik);
}

float bernstein(long i, long n, float t)
{
    double r = binomial(n, i);
    r *= std::pow(1 - t, n - i);
    r *= std::pow(t, i);
    return r;
}

glm::vec3 calc_pt(const std::vector<glm::vec3> &ctrl_pts, const double &t)
{
    glm::vec3 pt(0);
    for (size_t i = 0; i < ctrl_pts.size(); ++i)
    {
        const float bm = bernstein(i, ctrl_pts.size() - 1, t);
        pt += bm * ctrl_pts[i];
    }
    return pt;
}

std::vector<glm::vec3> generate_bez_line(const std::vector<glm::vec3> &ctrl_pts,
                                         const unsigned int &pts_num)
{
    std::vector<glm::vec3> curve_points;
    const auto ts = linspace(0, 1, pts_num);
    for (const auto &t : ts)
    {
        const glm::vec3 pt = calc_pt(ctrl_pts, t);
        curve_points.emplace_back(pt);
    }
    return curve_points;
}
std::vector<float> generate_bez_line(const std::vector<glm::vec3> &ctrl_pts,
                                     const unsigned int &pts_num, const glm::vec3 &color)
{
    const std::vector<glm::vec3> curve_points = generate_bez_line(ctrl_pts, pts_num);

    std::vector<float> data;
    for (size_t i = 0; i < curve_points.size(); ++i)
    {
        data.emplace_back(curve_points[i].x);
        data.emplace_back(curve_points[i].y);
        data.emplace_back(curve_points[i].z);

        data.emplace_back(color.x);
        data.emplace_back(color.y);
        data.emplace_back(color.z);
    }
    return data;
}

/*a: poddiagonala, b: glavna dijagonala, c: nad dijagonala, d: rhs*/
std::vector<double> thomas_algorithm(const std::vector<double> &a, const std::vector<double> &b,
                                     const std::vector<double> &c, const std::vector<double> &d)
{
    size_t N = d.size();
    std::vector<double> f(N, 0.0);
    // Create the temporary vectors
    // Note that this is inefficient as it is possible to call
    // this function many times. A better implementation would
    // pass these temporary matrices by non-const reference to
    // save excess allocation and deallocation
    std::vector<double> c_star(N, 0.0);
    std::vector<double> d_star(N, 0.0);

    // This updates the coefficients in the first row
    // Note that we should be checking for division by zero here
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    // Create the c_star and d_star coefficients in the forward sweep
    for (int i = 1; i < (int) N; i++)
    {
        double m = 1.0 / (b[i] - a[i - 1] * c_star[i - 1]);
        c_star[i] = c[i - 1] * m;
        d_star[i] = (d[i] - a[i - 1] * d_star[i - 1]) * m;
    }

    // This is the reverse sweep, used to update the solution vector f
    for (int i = N - 1; i-- > 0;)
    {
        f[i] = d_star[i] - c_star[i] * d[i + 1];
    }
    return f;
}

std::pair<std::vector<double>, std::vector<double>>
calc_interp_bez_coeffs(const std::vector<double> &interp_points /*, const unsigned int &pt_num*/)
{
    const unsigned int n = interp_points.size() - 1;
    std::vector<double> b(n, 4);
    b[0] = 2.;
    b[n - 1] = 7;
    std::vector<double> a(n - 1, 1);
    a[n - 2] = 2.;
    std::vector<double> c(n - 1, 1);

    std::vector<double> d(n);
    for (size_t i = 0; i < n; i++) d[i] = 2 * (2 * interp_points[i] + interp_points[i + 1]);

    d[0] = interp_points[0] + 2 * interp_points[1];
    d[n - 1] = 8 * interp_points[n - 1] + interp_points[n];

    const std::vector<double> A = thomas_algorithm(a, b, c, d);
    std::vector<double> B(n);
    for (size_t i = 0; i < n - 1; i++)
    {
        B[i] = 2 * interp_points[i + 1] - A[i + 1];
    }
    B[n - 1] = (A[n - 1] + interp_points[n]) / 2;

    return std::make_pair(A, B);
}

struct triangle {
    std::vector<float> data;
    // ctor
    triangle(std::vector<glm::vec3> v, std::vector<glm::vec3> c)
    {
        for (size_t i = 0; i < v.size(); i++)
        {
            data.emplace_back(v[i].x);
            data.emplace_back(v[i].y);
            data.emplace_back(v[i].z);
            data.emplace_back(c[i].x);
            data.emplace_back(c[i].y);
            data.emplace_back(c[i].z);
        }
    }
    void update_colors(std::vector<glm::vec3> c)
    {
        for (size_t i = 0; i < c.size(); i++)
        {
            data[i * 6 + 3] = c[i].x;
            data[i * 6 + 4] = c[i].y;
            data[i * 6 + 5] = c[i].z;
        }
    }
};

int main()
{
    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "cvetkovski", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(2);

    glewExperimental = true;// ovo nam treba za CORE Profile

    unsigned int err = glewInit();
    if (err != GLEW_OK)
    {
        std::cout << "error\n";
        return -1;
    }

    std::cout << glGetString(GL_VERSION) << std::endl;

    glEnable(GL_DEPTH_TEST);

    std::cout << "shaders folder: " << shaders_folder << std::endl;
    const std::string t_vs = shaders_folder + "vert.glsl";
    const std::string t_fs = shaders_folder + "frag.glsl";

    Shader bez_shader({{GL_VERTEX_SHADER, t_vs}, {GL_FRAGMENT_SHADER, t_fs}});

    //////////////////////////////////////////////////////////////////////////////////////
    // control points
    
    std::vector<glm::vec3> b00_pts{glm::vec3(-1.6, 1.15, 0.0), glm::vec3(-1., 0.8, 0), glm::vec3(-1., -1., 0.)};
    std::vector<glm::vec3> b01_pts{glm::vec3(-1., -1., 0.), glm::vec3(-0.6, 2., 0), glm::vec3(-0.3, -1., 0.)};
    std::vector<glm::vec3> b02_pts{glm::vec3(-0.3, -1., 0.), glm::vec3(0., 1.5, 0), glm::vec3(0.3, -1., 0.)};
    std::vector<glm::vec3> b03_pts{glm::vec3(0.3, -1., 0.), glm::vec3(0.6, 1., 0), glm::vec3(1., -1., 0.)};
    std::vector<glm::vec3> b04_pts{glm::vec3(1., -1., 0.), glm::vec3(1.3, 0.5, 0), glm::vec3(1.6, -1., 0.)};

    //prva krivulja
    std::vector<unsigned int> b00_indices{0, 1, 2};
    IndexBuffer b00_ib(b00_indices);
    std::vector<float> b00_data{
            b00_pts[0].x, b00_pts[0].y, b00_pts[0].z, 1., 0., 0.,
            b00_pts[1].x, b00_pts[1].y, b00_pts[1].z, 1., 0., 0.,
            b00_pts[2].x, b00_pts[2].y, b00_pts[2].z, 1., 0., 0.,
    };
    //druga krivulja
    std::vector<unsigned int> b01_indices{0, 1, 2};
    IndexBuffer b01_ib(b01_indices);
    std::vector<float> b01_data{
            b01_pts[0].x, b01_pts[0].y, b01_pts[0].z, 1., 0., 0.,
            b01_pts[1].x, b01_pts[1].y, b01_pts[1].z, 1., 0., 0.,
            b01_pts[2].x, b01_pts[2].y, b01_pts[2].z, 1., 0., 0.,
    };
    //treca krivulja
    std::vector<unsigned int> b02_indices{0, 1, 2};
    IndexBuffer b02_ib(b02_indices);
    std::vector<float> b02_data{
            b02_pts[0].x, b02_pts[0].y, b02_pts[0].z, 1., 0., 0.,
            b02_pts[1].x, b02_pts[1].y, b02_pts[1].z, 1., 0., 0.,
            b02_pts[2].x, b02_pts[2].y, b02_pts[2].z, 1., 0., 0.,
    };
    //cetvrta krivulja
    std::vector<unsigned int> b03_indices{0, 1, 2};
    IndexBuffer b03_ib(b03_indices);
    std::vector<float> b03_data{
            b03_pts[0].x, b03_pts[0].y, b03_pts[0].z, 1., 0., 0.,
            b03_pts[1].x, b03_pts[1].y, b03_pts[1].z, 1., 0., 0.,
            b03_pts[2].x, b03_pts[2].y, b03_pts[2].z, 1., 0., 0.,
    };
    //peta krivulja
    std::vector<unsigned int> b04_indices{0, 1, 2};
    IndexBuffer b04_ib(b04_indices);
    std::vector<float> b04_data{
            b04_pts[0].x, b04_pts[0].y, b04_pts[0].z, 1., 0., 0.,
            b04_pts[1].x, b04_pts[1].y, b04_pts[1].z, 1., 0., 0.,
            b04_pts[2].x, b04_pts[2].y, b04_pts[2].z, 1., 0., 0.,
    };

    //prva krivulja
    VertexBuffer b00_vb(b00_data);
    VertexBufferLayout b00_layout;
    b00_layout.addFloat(3);
    b00_layout.addFloat(3);

    VertexArray b00_va;
    b00_va.addBuffer(b00_vb, b00_layout);
    
    //druga krivulja
    VertexBuffer b01_vb(b01_data);
    VertexBufferLayout b01_layout;
    b01_layout.addFloat(3);
    b01_layout.addFloat(3);

    VertexArray b01_va;
    b01_va.addBuffer(b01_vb, b01_layout);

    //treca krivulja
    VertexBuffer b02_vb(b02_data);
    VertexBufferLayout b02_layout;
    b02_layout.addFloat(3);
    b02_layout.addFloat(3);

    VertexArray b02_va;
    b02_va.addBuffer(b02_vb, b02_layout);

    //cetvrta krivulja
    VertexBuffer b03_vb(b03_data);
    VertexBufferLayout b03_layout;
    b03_layout.addFloat(3);
    b03_layout.addFloat(3);

    VertexArray b03_va;
    b03_va.addBuffer(b03_vb, b03_layout);

    //peta krivulja
    VertexBuffer b04_vb(b04_data);
    VertexBufferLayout b04_layout;
    b04_layout.addFloat(3);
    b04_layout.addFloat(3);

    VertexArray b04_va;
    b04_va.addBuffer(b04_vb, b04_layout);
    
    // bezier line:
    //prva krivulja
    const unsigned int bez_pts0{20};
    const std::vector<float> bez_data0 = generate_bez_line(b00_pts, bez_pts0, glm::vec3(1, 0, 1));
    
    VertexBuffer bez_vb0(bez_data0);

    std::vector<unsigned int> bez_indices0(bez_pts0);
    std::iota(bez_indices0.begin(), bez_indices0.end(), 0);
    IndexBuffer bez_ib0(bez_indices0);

    VertexArray bez_va0;
    bez_va0.addBuffer(bez_vb0, b00_layout);

    //druga krivulja
    const unsigned int bez_pts1{20};
    const std::vector<float> bez_data1 = generate_bez_line(b01_pts, bez_pts1, glm::vec3(1, 0, 1));

    VertexBuffer bez_vb1(bez_data1);

    std::vector<unsigned int> bez_indices1(bez_pts1);
    std::iota(bez_indices1.begin(), bez_indices1.end(), 0);
    IndexBuffer bez_ib1(bez_indices1);

    VertexArray bez_va1;
    bez_va1.addBuffer(bez_vb1, b01_layout);
    
    //treca krivulja
    const unsigned int bez_pts2{20};
    const std::vector<float> bez_data2 = generate_bez_line(b02_pts, bez_pts2, glm::vec3(1, 0, 1));

    VertexBuffer bez_vb2(bez_data2);

    std::vector<unsigned int> bez_indices2(bez_pts2);
    std::iota(bez_indices2.begin(), bez_indices2.end(), 0);
    IndexBuffer bez_ib2(bez_indices2);

    VertexArray bez_va2;
    bez_va2.addBuffer(bez_vb2, b02_layout);

    //cetvrta krivulja
    const unsigned int bez_pts3{20};
    const std::vector<float> bez_data3 = generate_bez_line(b03_pts, bez_pts3, glm::vec3(1, 0, 1));

    VertexBuffer bez_vb3(bez_data3);

    std::vector<unsigned int> bez_indices3(bez_pts3);
    std::iota(bez_indices3.begin(), bez_indices3.end(), 0);
    IndexBuffer bez_ib3(bez_indices3);

    VertexArray bez_va3;
    bez_va3.addBuffer(bez_vb3, b03_layout);

    //peta krivulja
    const unsigned int bez_pts4{20};
    const std::vector<float> bez_data4 = generate_bez_line(b04_pts, bez_pts4, glm::vec3(1, 0, 1));

    VertexBuffer bez_vb4(bez_data4);

    std::vector<unsigned int> bez_indices4(bez_pts4);
    std::iota(bez_indices4.begin(), bez_indices4.end(), 0);
    IndexBuffer bez_ib4(bez_indices4);

    VertexArray bez_va4;
    bez_va4.addBuffer(bez_vb4, b04_layout);

    //prvi trail
    std::vector<glm::vec3> trail00_data{0};
    VertexBufferLayout trail00_layout;
    std::vector<unsigned int> trail00_indices{0};
    trail00_layout.addFloat(3);
    trail00_layout.addFloat(3);

    //drugi trail
    std::vector<glm::vec3> trail01_data;
    VertexBufferLayout trail01_layout;
    std::vector<unsigned int> trail01_indices{0};
    trail01_layout.addFloat(3);
    trail01_layout.addFloat(3);

    //treci trail
    std::vector<glm::vec3> trail02_data;
    VertexBufferLayout trail02_layout;
    std::vector<unsigned int> trail02_indices{0};
    trail02_layout.addFloat(3);
    trail02_layout.addFloat(3);

    //cetvrti trail
    std::vector<glm::vec3> trail03_data;
    VertexBufferLayout trail03_layout;
    std::vector<unsigned int> trail03_indices{0};
    trail03_layout.addFloat(3);
    trail03_layout.addFloat(3);

    //peti trail
    std::vector<glm::vec3> trail04_data;
    VertexBufferLayout trail04_layout;
    std::vector<unsigned int> trail04_indices{0};
    trail04_layout.addFloat(3);
    trail04_layout.addFloat(3);
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
    // trokut
    const std::vector<glm::vec3> t_colors = {
            glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)};

    std::vector<glm::vec3> t_vertices = {
            glm::vec3(-3., -1., 0),
            glm::vec3( 3., -1., 0),
            glm::vec3( 0, -3., 0),
    };
    triangle small_t(t_vertices, t_colors);

    std::vector<unsigned int> t_indices{0, 1, 2};
    IndexBuffer ib(t_indices);
    VertexBuffer small_vb(small_t.data);

    VertexBufferLayout t_layout;
    t_layout.addFloat(3);
    t_layout.addFloat(3);

    VertexArray small_va;
    small_va.addBuffer(small_vb, t_layout);

    //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
    // Sphere
    Texture sphere_map(images_folder + "blue-texture.jpg");
    const std::string sphere_vs = shaders_folder + "sphere_vert.glsl";
    const std::string sphere_fs = shaders_folder + "sphere_frag.glsl";
    Shader sphere_shader({{GL_VERTEX_SHADER, sphere_vs}, {GL_FRAGMENT_SHADER, sphere_fs}});
    sphere_shader.bind();
    // Sphere
    Sphere sphere(0.1f, 36, 18, true);

    VertexBuffer sphere_vb(sphere.getInterleavedVertices());
    VertexBufferLayout sphere_layout;
    sphere_layout.addFloat(3);// pos
    sphere_layout.addFloat(3);// normals
    sphere_layout.addFloat(2);// texture

    VertexArray sphere_va;
    sphere_va.addBuffer(sphere_vb, sphere_layout);
    IndexBuffer sphere_ib(sphere.getIndices());
    glm::mat4 m;

    const unsigned int bez_data_size = bez_data0.size();
    const unsigned int id_components_no = bez_data_size / 6;

    int istep = 1;
    int pos = 0;
    int krivulja = 0;
    bool prekidac = false;
    int brojac = 0;

    const glm::vec3 sphereColor(1., 0., 0.);
    const glm::vec3 lightColor(1., 1., 1.);

    while (!glfwWindowShouldClose(window))
    {
        _sleep(10);

        glClearColor(0.5f, 0.7f, 0.7f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const glm::vec3 viewPos(0.f, 0.0f, 4.f);
        const glm::vec3 lightPos(2, 2, 7);
        Renderer::clear();

        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);

        view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                           glm::vec3(0.0f, 1.0f, 0.0f));

        projection = glm::perspective(glm::radians(45.0f), (float) SCR_WIDTH / (float) SCR_HEIGHT,
                                      0.1f, 100.0f);

        bez_shader.bind();
        bez_shader.setMat4("view", view);
        bez_shader.setMat4("projection", projection);
        bez_shader.setMat4("transform", glm::mat4(1.0f));

        // support points
        /*Renderer::drawLines(b00_va, b00_ib, bez_shader);
        Renderer::drawLines(b01_va, b01_ib, bez_shader);
        Renderer::drawLines(b02_va, b02_ib, bez_shader);
        Renderer::drawLines(b03_va, b03_ib, bez_shader);
        Renderer::drawLines(b04_va, b04_ib, bez_shader);*/

        if (glfwGetKey(window, 83) == GLFW_PRESS)
        {
            prekidac = true;
        }

        if (glfwGetKey(window, 88) == GLFW_PRESS)
        {
            prekidac = false;
        }

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwDestroyWindow(window);
            break;
        }

        if (pos == 0 && krivulja == 0)
        {
            m = glm::translate(glm::mat4(1), glm::vec3(-1.6, 1.15, 0.0));
        }

        if (prekidac && (pos == 0 || pos == id_components_no))
        {
            if (krivulja < 5)
            {
                brojac = 0;
                krivulja++;
            }
            else
            {
                krivulja = 0;

                trail00_indices.clear();
                trail01_indices.clear();
                trail02_indices.clear();
                trail03_indices.clear();
                trail04_indices.clear();
            }
        }

        if (krivulja == 1 && prekidac)
        {
            //////////////////////
            trail00_data.emplace_back(glm::vec3(bez_data0[pos * 6], 
                                                bez_data0[pos * 6 + 1],
                                                bez_data0[pos * 6 + 2]));
            trail00_data.emplace_back(glm::vec3(1, 0, 0));
            trail00_indices.emplace_back(brojac);
            brojac++;
            ////////////////////////
            
            //Renderer::drawLines(bez_va0, bez_ib0, bez_shader);

            m = glm::translate(glm::mat4(1), glm::vec3(bez_data0[pos * 6], 
                                                       bez_data0[pos * 6 + 1],
                                                       bez_data0[pos * 6 + 2]));
        }

        if (krivulja > 0)
        {
            VertexBuffer trail00_vb(trail00_data);
            VertexArray trail00_va;
            
            trail00_va.addBuffer(trail00_vb, trail00_layout);

                    IndexBuffer trail00_ib(trail00_indices);

                    Renderer::drawLines(trail00_va, trail00_ib, bez_shader);
                }
                
        if (krivulja == 2 && prekidac)
        {
            //////////////////////
            trail01_data.emplace_back(glm::vec3(bez_data1[pos * 6], 
                                                bez_data1[pos * 6 + 1],
                                                bez_data1[pos * 6 + 2]));
            trail01_data.emplace_back(glm::vec3(1, 0, 0));
            
            trail01_indices.emplace_back(brojac);
            brojac++;
            ////////////////////////
            
            //Renderer::drawLines(bez_va1, bez_ib1, bez_shader);
            m = glm::translate(glm::mat4(1),glm::vec3(bez_data1[pos * 6], 
                                                      bez_data1[pos * 6 + 1],
                                                      bez_data1[pos * 6 + 2]));
        }

        if (krivulja > 1)
        {
             VertexBuffer trail01_vb(trail01_data);

             VertexArray trail01_va;
             trail01_va.addBuffer(trail01_vb, trail01_layout);

             IndexBuffer trail01_ib(trail01_indices);

             Renderer::drawLines(trail01_va, trail01_ib, bez_shader);
        }
                
        if (krivulja == 3 && prekidac)
        {
             //////////////////////
             trail02_data.emplace_back(glm::vec3(bez_data2[pos * 6], 
                                                 bez_data2[pos * 6 + 1],
                                                 bez_data2[pos * 6 + 2]));
             trail02_data.emplace_back(glm::vec3(1, 0, 0));

             trail02_indices.emplace_back(brojac);
             brojac++;
             ////////////////////////

             //Renderer::drawLines(bez_va2, bez_ib2, bez_shader);
             m = glm::translate(glm::mat4(1), glm::vec3(bez_data2[pos * 6], 
                                                        bez_data2[pos * 6 + 1],
                                                        bez_data2[pos * 6 + 2]));
        }

        if (krivulja > 2)
        {
            VertexBuffer trail02_vb(trail02_data);

            VertexArray trail02_va;
            trail02_va.addBuffer(trail02_vb, trail02_layout);

            IndexBuffer trail02_ib(trail02_indices);

            Renderer::drawLines(trail02_va, trail02_ib, bez_shader);
        }
                
        if (krivulja == 4 && prekidac)
        {
            //////////////////////
            trail03_data.emplace_back(glm::vec3(bez_data3[pos * 6], 
                                                bez_data3[pos * 6 + 1],
                                                bez_data3[pos * 6 + 2]));
            trail03_data.emplace_back(glm::vec3(1, 0, 0));

            trail03_indices.emplace_back(brojac);
            brojac++;
            ////////////////////////

            //Renderer::drawLines(bez_va3, bez_ib3, bez_shader);
            m = glm::translate(glm::mat4(1), glm::vec3(bez_data3[pos * 6], 
                                                    bez_data3[pos * 6 + 1],
                                                    bez_data3[pos * 6 + 2]));
        }

        if (krivulja > 3)
        {
            VertexBuffer trail03_vb(trail03_data);

            VertexArray trail03_va;
            trail03_va.addBuffer(trail03_vb, trail03_layout);

            IndexBuffer trail03_ib(trail03_indices);

            Renderer::drawLines(trail03_va, trail03_ib, bez_shader);
        }

        if (krivulja == 5 && prekidac)
        {
            //////////////////////
            trail04_data.emplace_back(glm::vec3(bez_data4[pos * 6], 
                                                bez_data4[pos * 6 + 1],
                                                bez_data4[pos * 6 + 2]));
            trail04_data.emplace_back(glm::vec3(1, 0, 0));
                    
            trail04_indices.emplace_back(brojac);
            brojac++;
            ////////////////////////

            //Renderer::drawLines(bez_va4, bez_ib4, bez_shader);
            m = glm::translate(glm::mat4(1), glm::vec3(bez_data4[pos * 6], 
                                                       bez_data4[pos * 6 + 1],
                                                       bez_data4[pos * 6 + 2]));
        }

        if (krivulja > 4)
        {
            VertexBuffer trail04_vb(trail04_data);

            VertexArray trail04_va;
            trail04_va.addBuffer(trail04_vb, trail04_layout);

            IndexBuffer trail04_ib(trail04_indices);
                    
            Renderer::drawLines(trail04_va, trail04_ib, bez_shader);
        }

        if (prekidac)
        {
            pos += istep;

            if (pos >= id_components_no || pos <= 0)
            {
                if (pos == id_components_no && krivulja == 5)
                {
                    prekidac = false;
                }
                else
                pos = 0;
            }
        }
            
        // sphere & cylinder
        sphere_shader.bind();
        sphere_map.bind(0);
        sphere_shader.setMat4("model", m /*glm::mat4(1)*/);
        sphere_shader.setMat4("view", view);
        sphere_shader.setMat4("proj", projection);

        sphere_shader.setVec3("objectColor", sphereColor);
        sphere_shader.setVec3("lightColor", lightColor);
        sphere_shader.setVec3("lightPos", lightPos);
        sphere_shader.setVec3("viewPos", viewPos);

        // draw sphere
        Renderer::drawTriangles(sphere_va, sphere_ib, sphere_shader);

        // trokut
        bez_shader.setMat4("transform", m);
        Renderer::drawTriangles(small_va, ib, bez_shader);
       
        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
    
    glfwTerminate();
    return 0;
}
