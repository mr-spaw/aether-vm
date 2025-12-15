#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <GL/glut.h>

// ============================================================
// Physical constants (SI)
// ============================================================
constexpr double EPS0 = 8.854187817e-12;
constexpr double C    = 2.99792458e8;
constexpr double QE   = -1.60217662e-19;
constexpr double ME   = 9.10938356e-31;
constexpr double MP   = 1.672621898e-27;

// ============================================================
// Simulation parameters
// ============================================================
constexpr int    NX = 80, NY = 80, NZ = 80;
constexpr int    NP = 60000;
constexpr double DX = 5e-4;
constexpr double DT = 2e-14;
constexpr double L  = NX * DX;
constexpr int    SUBSTEP = 6;

// ============================================================
// Visualization settings
// ============================================================
bool showParticles = true;
bool showElectrons = true;
bool showIons = true;
bool showEField = true;
bool showBField = false;
bool showDensity = true;
int renderMode = 0; // 0=all, 1=slice, 2=volume

// ============================================================
// Camera controls
// ============================================================
float camDist = 3.5f;
float camAngleX = 25.0f;
float camAngleY = 45.0f;
int mouseX = 0, mouseY = 0;
bool mouseDown = false;
float autoRotate = 0.0f;

// ============================================================
// Particle structure
// ============================================================
struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double q, m;
    int species;
};

// ============================================================
// Global simulation data
// ============================================================
std::vector<Particle> particles;
std::vector<double> Ex, Ey, Ez, Bx, By, Bz;
std::vector<double> rho, Jx, Jy, Jz;
std::vector<double> density;

int idx(int i, int j, int k) {
    i = (i + NX) % NX;
    j = (j + NY) % NY;
    k = (k + NZ) % NZ;
    return i + NX * (j + NY * k);
}

// ============================================================
// Physics routines
// ============================================================
void depositCharge() {
    std::fill(rho.begin(), rho.end(), 0.0);
    std::fill(density.begin(), density.end(), 0.0);

    for (auto& p : particles) {
        int i = int(p.x / DX);
        int j = int(p.y / DX);
        int k = int(p.z / DX);
        
        if (i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ) {
            int id = idx(i, j, k);
            rho[id] += p.q / (DX * DX * DX);
            density[id] += 1.0;
        }
    }
    double ion_rho = -(QE * (NP/2)) / (L*L*L);
    for (double &r : rho) r += ion_rho;

}

void depositCurrent() {
    std::fill(Jx.begin(), Jx.end(), 0.0);
    std::fill(Jy.begin(), Jy.end(), 0.0);
    std::fill(Jz.begin(), Jz.end(), 0.0);

    for (auto& p : particles) {
        int i = int(p.x / DX);
        int j = int(p.y / DX);
        int k = int(p.z / DX);
        
        if (i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ) {
            int id = idx(i, j, k);
            double vol = DX * DX * DX;
            Jx[id] += p.q * p.vx / vol;
            Jy[id] += p.q * p.vy / vol;
            Jz[id] += p.q * p.vz / vol;
        }
    }
}
void updateFields() {
    const double EMAX = 1e6;

    // -------------------------
    // Faraday's law (∂B/∂t = -∇×E)
    // -------------------------
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int id = idx(i, j, k);

                double curlEx_y = (Ez[idx(i, j+1, k)] - Ez[id]) / DX;
                double curlEx_z = (Ey[idx(i, j, k+1)] - Ey[id]) / DX;
                double curlEy_z = (Ex[idx(i, j, k+1)] - Ex[id]) / DX;
                double curlEy_x = (Ez[idx(i+1, j, k)] - Ez[id]) / DX;
                double curlEz_x = (Ey[idx(i+1, j, k)] - Ey[id]) / DX;
                double curlEz_y = (Ex[idx(i, j+1, k)] - Ex[id]) / DX;

                Bx[id] -= DT * (curlEx_y - curlEx_z);
                By[id] -= DT * (curlEy_z - curlEy_x);
                Bz[id] -= DT * (curlEz_x - curlEz_y);
            }
        }
    }

    // -------------------------
    // Ampere-Maxwell law (∂E/∂t = c²∇×B − J/ε₀)
    // -------------------------
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int id = idx(i, j, k);

                double curlBx_y = (Bz[idx(i, j+1, k)] - Bz[id]) / DX;
                double curlBx_z = (By[idx(i, j, k+1)] - By[id]) / DX;
                double curlBy_z = (Bx[idx(i, j, k+1)] - Bx[id]) / DX;
                double curlBy_x = (Bz[idx(i+1, j, k)] - Bz[id]) / DX;
                double curlBz_x = (By[idx(i+1, j, k)] - By[id]) / DX;
                double curlBz_y = (Bx[idx(i, j+1, k)] - Bx[id]) / DX;

                Ex[id] += DT * (C * C * (curlBx_y - curlBx_z) - Jx[id] / EPS0);
                Ey[id] += DT * (C * C * (curlBy_z - curlBy_x) - Jy[id] / EPS0);
                Ez[id] += DT * (C * C * (curlBz_x - curlBz_y) - Jz[id] / EPS0);

                // -------------------------
                // SAFETY CLAMP (debug only)
                // -------------------------
                Ex[id] = std::max(-EMAX, std::min(EMAX, Ex[id]));
                Ey[id] = std::max(-EMAX, std::min(EMAX, Ey[id]));
                Ez[id] = std::max(-EMAX, std::min(EMAX, Ez[id]));
            }
        }
    }
}

void pushParticles() {
    for (auto& p : particles) {
        int i = int(p.x / DX);
        int j = int(p.y / DX);
        int k = int(p.z / DX);
        
        if (i < 0 || i >= NX || j < 0 || j >= NY || k < 0 || k >= NZ) continue;
        
        int id = idx(i, j, k);
        
        double ex = Ex[id];
        double ey = Ey[id];
        double ez = Ez[id];
        double bx = Bx[id];
        double by = By[id];
        double bz = Bz[id];
        
        double qm = p.q / p.m;
        double dt_half = 0.5 * DT;
        
        // Boris pusher
        p.vx += qm * ex * dt_half;
        p.vy += qm * ey * dt_half;
        p.vz += qm * ez * dt_half;
        
        double t_x = qm * bx * dt_half;
        double t_y = qm * by * dt_half;
        double t_z = qm * bz * dt_half;
        double t2 = t_x * t_x + t_y * t_y + t_z * t_z;
        double s_fac = 2.0 / (1.0 + t2);
        
        double v_minus_x = p.vx;
        double v_minus_y = p.vy;
        double v_minus_z = p.vz;
        
        double v_prime_x = v_minus_x + v_minus_y * t_z - v_minus_z * t_y;
        double v_prime_y = v_minus_y + v_minus_z * t_x - v_minus_x * t_z;
        double v_prime_z = v_minus_z + v_minus_x * t_y - v_minus_y * t_x;
        
        p.vx = v_minus_x + s_fac * (v_prime_y * t_z - v_prime_z * t_y);
        p.vy = v_minus_y + s_fac * (v_prime_z * t_x - v_prime_x * t_z);
        p.vz = v_minus_z + s_fac * (v_prime_x * t_y - v_prime_y * t_x);
        
        p.vx += qm * ex * dt_half;
        p.vy += qm * ey * dt_half;
        p.vz += qm * ez * dt_half;
        
        // Velocity limiter
        double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        double vmax = 0.1 * C;
        if (v2 > vmax*vmax) {
            double scale = vmax / sqrt(v2);
            p.vx *= scale;
            p.vy *= scale;
            p.vz *= scale;
        }
        
        p.x += p.vx * DT;
        p.y += p.vy * DT;
        p.z += p.vz * DT;
        
        // Periodic boundaries
        if (p.x < 0) p.x += L;
        if (p.x >= L) p.x -= L;
        if (p.y < 0) p.y += L;
        if (p.y >= L) p.y -= L;
        if (p.z < 0) p.z += L;
        if (p.z >= L) p.z -= L;
    }
}

// ============================================================
// Advanced OpenGL visualization
// ============================================================
void drawDensityVolume() {
    if (!showDensity) return;
    
    double maxDens = 1e-30;
    for (double d : density) maxDens = std::max(maxDens, d);
    if (maxDens < 1) return;
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    
    int skip = 2;
    for (int i = 0; i < NX; i += skip) {
        for (int j = 0; j < NY; j += skip) {
            for (int k = 0; k < NZ; k += skip) {
                int id = idx(i, j, k);
                double dens = density[id] / maxDens;
                
                if (dens < 0.05) continue;
                
                float x = 2.0f * (i * DX / L - 0.5f);
                float y = 2.0f * (j * DX / L - 0.5f);
                float z = 2.0f * (k * DX / L - 0.5f);
                
                float size = 0.03f;
                float alpha = 0.15f * dens;
                
                // Heat map coloring
                float r = std::min<float>(1.0f, 2.0f * dens);
                float g = std::min<float>(1.0f, 2.0f * (1.0f - dens));
                float b = 0.3f;
                
                glColor4f(r, g, b, alpha);
                
                glBegin(GL_QUADS);
                // Draw cube
                glVertex3f(x-size, y-size, z-size);
                glVertex3f(x+size, y-size, z-size);
                glVertex3f(x+size, y+size, z-size);
                glVertex3f(x-size, y+size, z-size);
                
                glVertex3f(x-size, y-size, z+size);
                glVertex3f(x+size, y-size, z+size);
                glVertex3f(x+size, y+size, z+size);
                glVertex3f(x-size, y+size, z+size);
                glEnd();
            }
        }
    }
    
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void drawParticlesEnhanced() {
    if (!showParticles) return;
    
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    
    // Draw ions (larger, glowing)
    if (showIons) {
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        for (auto& p : particles) {
            if (p.species != 1) continue;
            
            float x = 2.0f * (p.x / L - 0.5f);
            float y = 2.0f * (p.y / L - 0.5f);
            float z = 2.0f * (p.z / L - 0.5f);
            
            double speed = sqrt(p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
            float intensity = std::min<float>(1.0f, speed / 1e6);
            
            glColor4f(1.0f, 0.2f + 0.5f*intensity, 0.1f, 0.9f);
            glVertex3f(x, y, z);
        }
        glEnd();
    }
    
    // Draw electrons (smaller, fast)
    if (showElectrons) {
        glPointSize(2.5f);
        glBegin(GL_POINTS);
        for (auto& p : particles) {
            if (p.species != 0) continue;
            
            float x = 2.0f * (p.x / L - 0.5f);
            float y = 2.0f * (p.y / L - 0.5f);
            float z = 2.0f * (p.z / L - 0.5f);
            
            double speed = sqrt(p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
            float intensity = std::min<float>(1.0f, speed / 1e7);
            
            glColor4f(0.2f, 0.5f + 0.5f*intensity, 1.0f, 0.8f);
            glVertex3f(x, y, z);
        }
        glEnd();
    }
}

void drawElectricFieldEnhanced() {
    if (!showEField) return;
    
    double maxE = 1e-30;
    for (double e : Ex) maxE = std::max(maxE, std::abs(e));
    for (double e : Ey) maxE = std::max(maxE, std::abs(e));
    for (double e : Ez) maxE = std::max(maxE, std::abs(e));
    
    if (maxE < 1e-5) return;
    
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    
    int skip = 3;
    for (int i = 0; i < NX; i += skip) {
        for (int j = 0; j < NY; j += skip) {
            for (int k = 0; k < NZ; k += skip) {
                int id = idx(i, j, k);
                
                double ex = Ex[id] / maxE;
                double ey = Ey[id] / maxE;
                double ez = Ez[id] / maxE;
                
                double emag = sqrt(ex*ex + ey*ey + ez*ez);
                if (emag < 0.15) continue;
                
                float x = 2.0f * (i * DX / L - 0.5f);
                float y = 2.0f * (j * DX / L - 0.5f);
                float z = 2.0f * (k * DX / L - 0.5f);
                
                float scale = 0.15f;
                float dx = scale * ex;
                float dy = scale * ey;
                float dz = scale * ez;
                
                float intensity = std::min(1.0f, (float)emag);
                glColor4f(0.0f, 1.0f, 0.2f, 0.7f * intensity);
                
                glVertex3f(x - dx/2, y - dy/2, z - dz/2);
                glVertex3f(x + dx/2, y + dy/2, z + dz/2);
            }
        }
    }
    
    glEnd();
}

void drawMagneticField() {
    if (!showBField) return;
    
    double maxB = 1e-30;
    for (double b : Bx) maxB = std::max(maxB, std::abs(b));
    for (double b : By) maxB = std::max(maxB, std::abs(b));
    for (double b : Bz) maxB = std::max(maxB, std::abs(b));
    
    if (maxB < 1e-10) return;
    
    glLineWidth(1.5f);
    glBegin(GL_LINES);
    
    int skip = 4;
    for (int i = 0; i < NX; i += skip) {
        for (int j = 0; j < NY; j += skip) {
            for (int k = 0; k < NZ; k += skip) {
                int id = idx(i, j, k);
                
                double bx = Bx[id] / maxB;
                double by = By[id] / maxB;
                double bz = Bz[id] / maxB;
                
                double bmag = sqrt(bx*bx + by*by + bz*bz);
                if (bmag < 0.2) continue;
                
                float x = 2.0f * (i * DX / L - 0.5f);
                float y = 2.0f * (j * DX / L - 0.5f);
                float z = 2.0f * (k * DX / L - 0.5f);
                
                float scale = 0.12f;
                float dx = scale * bx;
                float dy = scale * by;
                float dz = scale * bz;
                
                float intensity = std::min(1.0f, (float)bmag);
                glColor4f(1.0f, 0.0f, 1.0f, 0.6f * intensity);
                
                glVertex3f(x - dx/2, y - dy/2, z - dz/2);
                glVertex3f(x + dx/2, y + dy/2, z + dz/2);
            }
        }
    }
    
    glEnd();
}

void drawBoundingBox() {
    glColor4f(0.5f, 0.5f, 0.5f, 0.3f);
    glLineWidth(2.0f);
    
    glBegin(GL_LINE_LOOP);
    glVertex3f(-1, -1, -1);
    glVertex3f( 1, -1, -1);
    glVertex3f( 1,  1, -1);
    glVertex3f(-1,  1, -1);
    glEnd();
    
    glBegin(GL_LINE_LOOP);
    glVertex3f(-1, -1, 1);
    glVertex3f( 1, -1, 1);
    glVertex3f( 1,  1, 1);
    glVertex3f(-1,  1, 1);
    glEnd();
    
    glBegin(GL_LINES);
    glVertex3f(-1, -1, -1); glVertex3f(-1, -1, 1);
    glVertex3f( 1, -1, -1); glVertex3f( 1, -1, 1);
    glVertex3f( 1,  1, -1); glVertex3f( 1,  1, 1);
    glVertex3f(-1,  1, -1); glVertex3f(-1,  1, 1);
    glEnd();
}

void drawHUD() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 1000, 0, 1000);
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    glDisable(GL_DEPTH_TEST);
    
    // Draw text info
    glColor3f(0.0f, 1.0f, 0.0f);
    glRasterPos2f(10, 970);
    
    std::string info = "3D PLASMA FUSION SIMULATION | Particles: " + std::to_string(NP);
    for (char c : info) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    }
    
    glRasterPos2f(10, 950);
    std::string controls = "[E]Field [B]Magnetic [P]Particles [D]Density [R]Auto-rotate [SPACE]Reset";
    for (char c : controls) {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, c);
    }
    
    glEnable(GL_DEPTH_TEST);
    
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glTranslatef(0, 0, -camDist);
    glRotatef(camAngleX, 1, 0, 0);
    glRotatef(camAngleY + autoRotate, 0, 1, 0);
    
    drawBoundingBox();
    drawDensityVolume();
    drawMagneticField();
    drawElectricFieldEnhanced();
    drawParticlesEnhanced();
    
    drawHUD();
    
    glutSwapBuffers();
}

// ============================================================
// Time stepping loop
// ============================================================
void idle() {
    for (int s = 0; s < SUBSTEP; s++) {
        depositCharge();
        depositCurrent();
        updateFields();
        pushParticles();
    }
    
    static int counter = 0;
    if (++counter % 500 == 0) {
        double maxE = 0;
        for (double e : Ex) maxE = std::max(maxE, std::abs(e));
        for (double e : Ey) maxE = std::max(maxE, std::abs(e));
        for (double e : Ez) maxE = std::max(maxE, std::abs(e));
        
        double totalKE = 0;
        for (auto& p : particles) {
            double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
            totalKE += 0.5 * p.m * v2;
        }
        
        std::cout << "max|E|=" << maxE << " KE=" << totalKE << std::endl;
    }
    
    if (autoRotate != 0) {
        // Auto rotation active
    }
    
    glutPostRedisplay();
}

// ============================================================
// Input controls
// ============================================================
void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case 'e': case 'E':
            showEField = !showEField;
            std::cout << "Electric field: " << (showEField ? "ON" : "OFF") << std::endl;
            break;
        case 'b': case 'B':
            showBField = !showBField;
            std::cout << "Magnetic field: " << (showBField ? "ON" : "OFF") << std::endl;
            break;
        case 'p': case 'P':
            showParticles = !showParticles;
            std::cout << "Particles: " << (showParticles ? "ON" : "OFF") << std::endl;
            break;
        case 'd': case 'D':
            showDensity = !showDensity;
            std::cout << "Density: " << (showDensity ? "ON" : "OFF") << std::endl;
            break;
        case 'r': case 'R':
            autoRotate = (autoRotate == 0) ? 0.5f : 0.0f;
            std::cout << "Auto-rotate: " << (autoRotate != 0 ? "ON" : "OFF") << std::endl;
            break;
        case '1':
            showElectrons = !showElectrons;
            std::cout << "Electrons: " << (showElectrons ? "ON" : "OFF") << std::endl;
            break;
        case '2':
            showIons = !showIons;
            std::cout << "Ions: " << (showIons ? "ON" : "OFF") << std::endl;
            break;
        case ' ':
            std::cout << "Resetting simulation..." << std::endl;
            exit(0);
        case 27: // ESC
            exit(0);
            break;
    }
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        mouseDown = (state == GLUT_DOWN);
        mouseX = x;
        mouseY = y;
    }
    
    if (button == 3) camDist *= 0.9f;
    if (button == 4) camDist *= 1.1f;
    
    if (camDist < 1.0f) camDist = 1.0f;
    if (camDist > 10.0f) camDist = 10.0f;
}

void motion(int x, int y) {
    if (mouseDown) {
        camAngleY += (x - mouseX) * 0.5f;
        camAngleX += (y - mouseY) * 0.5f;
        
        if (camAngleX > 89) camAngleX = 89;
        if (camAngleX < -89) camAngleX = -89;
        
        mouseX = x;
        mouseY = y;
    }
}

// ============================================================
// Initialization
// ============================================================
void initSimulation() {
    int totalCells = NX * NY * NZ;
    
    Ex.resize(totalCells, 0.0);
    Ey.resize(totalCells, 0.0);
    Ez.resize(totalCells, 0.0);
    Bx.resize(totalCells, 0.0);
    By.resize(totalCells, 0.0);
    Bz.resize(totalCells, 0.0);
    rho.resize(totalCells, 0.0);
    Jx.resize(totalCells, 0.0);
    Jy.resize(totalCells, 0.0);
    Jz.resize(totalCells, 0.0);
    density.resize(totalCells, 0.0);
    
    particles.reserve(NP);
    
    // Create two colliding plasma beams
    for (int i = 0; i < NP / 2; i++) {
        // Beam 1: electrons and ions moving right
        Particle e1, ion1;
        
        double x = L * 0.25 + L * 0.1 * (rand() / double(RAND_MAX));
        double y = L * (rand() / double(RAND_MAX));
        double z = L * (rand() / double(RAND_MAX));
        
        e1.x = ion1.x = x;
        e1.y = ion1.y = y;
        e1.z = ion1.z = z;
        
        double vbeam = 5e6;
        e1.vx = vbeam + 1e6 * (rand() / double(RAND_MAX) - 0.5);
        e1.vy = 1e5 * (rand() / double(RAND_MAX) - 0.5);
        e1.vz = 1e5 * (rand() / double(RAND_MAX) - 0.5);
        
        ion1.vx = vbeam * 0.01 + 1e4 * (rand() / double(RAND_MAX) - 0.5);
        ion1.vy = 1e3 * (rand() / double(RAND_MAX) - 0.5);
        ion1.vz = 1e3 * (rand() / double(RAND_MAX) - 0.5);
        
        e1.q = QE;
        e1.m = ME;
        e1.species = 0;
        
        ion1.q = -QE;
        ion1.m = MP;
        ion1.species = 1;
        
        particles.push_back(e1);
        particles.push_back(ion1);
    }
    
    // Beam 2: electrons and ions moving left
    for (int i = 0; i < NP / 2; i++) {
        Particle e2, ion2;
        
        double x = L * 0.75 + L * 0.1 * (rand() / double(RAND_MAX) - 0.5);
        double y = L * (rand() / double(RAND_MAX));
        double z = L * (rand() / double(RAND_MAX));
        
        e2.x = ion2.x = x;
        e2.y = ion2.y = y;
        e2.z = ion2.z = z;
        
        double vbeam = -5e6;
        e2.vx = vbeam + 1e6 * (rand() / double(RAND_MAX) - 0.5);
        e2.vy = 1e5 * (rand() / double(RAND_MAX) - 0.5);
        e2.vz = 1e5 * (rand() / double(RAND_MAX) - 0.5);
        
        ion2.vx = vbeam * 0.01 + 1e4 * (rand() / double(RAND_MAX) - 0.5);
        ion2.vy = 1e3 * (rand() / double(RAND_MAX) - 0.5);
        ion2.vz = 1e3 * (rand() / double(RAND_MAX) - 0.5);
        
        e2.q = QE;
        e2.m = ME;
        e2.species = 0;
        
        ion2.q = -QE;
        ion2.m = MP;
        ion2.species = 1;
        
        particles.push_back(e2);
        particles.push_back(ion2);
    }
    
    // Seed strong electric field perturbation
    for (int i = NX/3; i < 2*NX/3; i++) {
        for (int j = NY/3; j < 2*NY/3; j++) {
            for (int k = NZ/3; k < 2*NZ/3; k++) {
                Ex[idx(i, j, k)] = 1e3 * cos(4 * M_PI * i / NX);
                Ey[idx(i, j, k)] = 1e3 * sin(4 * M_PI * j / NY);
                Ez[idx(i, j, k)] = 5e2 * sin(4 * M_PI * k / NZ);
            }
        }
    }
    
    std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   3D PLASMA FUSION PIC SIMULATION - INITIALIZED   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nGrid: " << NX << "×" << NY << "×" << NZ << " = " << (NX*NY*NZ) << " cells" << std::endl;
    std::cout << "Particles: " << NP << " (" << NP/2 << " electrons + " << NP/2 << " ions)" << std::endl;
    std::cout << "Domain size: " << L*1e3 << " mm³" << std::endl;
    std::cout << "\n━━━━━━━━━━━━━━━ CONTROLS ━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "[E] Toggle Electric Field Vectors" << std::endl;
    std::cout << "[B] Toggle Magnetic Field Vectors" << std::endl;
    std::cout << "[P] Toggle All Particles" << std::endl;
    std::cout << "[1] Toggle Electrons Only" << std::endl;
    std::cout << "[2] Toggle Ions Only" << std::endl;
    std::cout << "[D] Toggle Density Volume Rendering" << std::endl;
    std::cout << "[R] Toggle Auto-Rotation" << std::endl;
    std::cout << "Mouse: Left-click + drag to rotate" << std::endl;
    std::cout << "       Scroll to zoom" << std::endl;
    std::cout << "[ESC] Exit simulation" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
}

void initOpenGL() {
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50.0, 1.0, 0.1, 100.0);
    
    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 2.0f);
    glFogf(GL_FOG_END, 8.0f);
    GLfloat fogColor[] = {0.02f, 0.02f, 0.05f, 1.0f};
    glFogfv(GL_FOG_COLOR, fogColor);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    srand(42);
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1200, 1200);
    glutCreateWindow("3D Plasma Fusion PIC — Self-Consistent EM Fields");
    
    initSimulation();
    initOpenGL();
    
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    
    glutMainLoop();
    return 0;
}