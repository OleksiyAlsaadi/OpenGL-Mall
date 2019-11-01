#define GL_GLEXT_PROTOTYPES

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "SOIL.h"

#include <time.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace std;
int WIDTH = 800;
int HEIGHT = 800;

static const char* vertex_source = 
    "   #version 130 \n" 

    "   uniform vec4 u_Translation; \n"
    " 	uniform mat4 u_ViewMatrix; \n"
  	"	uniform mat4 u_ProjMatrix; \n"
  	"	uniform mat4 u_ModelMatrix; \n"
  	"	uniform mat4 u_NormalMatrix; \n"

    "   in vec4 a_Position; \n" 
    "   in vec4 a_Normal; \n" 
    "   in vec4 a_Color; \n"
    "   in vec2 a_TexCoord; \n"
    "   out vec4 v_Color; \n"

    "   varying vec2 v_TexCoord;\n"

    "   void main() { \n" 
    "		vec3 LightDir = vec3(1,1,1); \n"
    "		vec3 normal = normalize(vec3(u_NormalMatrix)); \n"
	"		float nDot = max(dot(normal, LightDir), 0.0); \n"

    "       gl_Position = a_Position * u_ModelMatrix * u_ViewMatrix * u_ProjMatrix;  \n" 
    "       v_Color = a_Color; \n"
    "       v_TexCoord = a_TexCoord;\n"
    "   } \n";

static const char* fragment_source =
    "   #version 130 \n"
    "   in vec4 v_Color; \n"

    "	uniform sampler2D u_Sampler;\n"
    "	varying vec2 v_TexCoord;\n"
    "	uniform float u_Time; \n"
    "   uniform float u_Ambient; \n"

    "   void main(){ \n"
    "		float cx = v_TexCoord.x; \n"
    "		float cy = v_TexCoord.y; \n"

    "       vec4 color = texture2D(u_Sampler, v_TexCoord); \n"
    "       gl_FragColor = v_Color * vec4(color.rgb, color.a); \n"
    // "       gl_FragColor = v_Color * vec4(color.rgb, color.a) * u_Ambient; \n"
    //"       gl_FragColor = gl_FragColor + vec4( m,m,m, 1.0); \n"
    "   }\n";


typedef enum {
    a_Position,
    a_Normal,
    a_Color,
    a_TexCoord,
} attrib_id;

struct Matrix4 {
    float elements[16] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};

	void setElements(float (&e)[16]){
		for (int n = 0; n < 16; n++){
            elements[n] = e[n];
        }
	}

    //Set default matrix
    void setIdentity(){
        float e[16];
        e[0] = 1;   e[4] = 0;   e[8]  = 0;   e[12] = 0;
        e[1] = 0;   e[5] = 1;   e[9]  = 0;   e[13] = 0;
        e[2] = 0;   e[6] = 0;   e[10] = 1;   e[14] = 0;
        e[3] = 0;   e[7] = 0;   e[11] = 0;   e[15] = 1;
        setElements(e);
    }

    //Copies Matrix to another
    void copyFrom(Matrix4 old){
        for(int n = 0; n < 16; n++){
            elements[n] = old.elements[n]; //16-n?
        }
    }

    //Print Matrix
    void print(){
        for(int n = 0; n < 16; n++){
            cout << elements[n] << " ";
            if (n%4 == 3) cout << endl;
        }
        cout << endl;
    }

    //Reverse matrix
	void transpose(){
		float t;
		float e[16];
		for (int n = 0; n < 16; n++){
            e[n] = elements[n];
        }
		t = e[ 1];  e[ 1] = e[ 4];  e[ 4] = t;
		t = e[ 2];  e[ 2] = e[ 8];  e[ 8] = t;
		t = e[ 3];  e[ 3] = e[12];  e[12] = t;
		t = e[ 6];  e[ 6] = e[ 9];  e[ 9] = t;
		t = e[ 7];  e[ 7] = e[13];  e[13] = t;
		t = e[11];  e[11] = e[14];  e[14] = t;
		setElements(e);
	}

	//SetTranslate on Translation matrix
	void setTranslate(float x, float y, float z) {
		float e[16];
		for (int n = 0; n < 16; n++){
            e[n] = elements[n];
        }
		e[0] = 1;  e[4] = 0;  e[8]  = 0;  e[12] = x;
		e[1] = 0;  e[5] = 1;  e[9]  = 0;  e[13] = y;
		e[2] = 0;  e[6] = 0;  e[10] = 1;  e[14] = z;
		e[3] = 0;  e[7] = 0;  e[11] = 0;  e[15] = 1;
		setElements(e);
		return;
	};

	//Translate matrix by x, y, z - multiply by x, y, z
	void translate(float x, float y, float z) {
		float e[16]; 
  		for (int n = 0; n < 16; n++){
            e[n] = elements[n];
        }
		e[12] += e[0] * x + e[4] * y + e[8]  * z;
		e[13] += e[1] * x + e[5] * y + e[9]  * z;
		e[14] += e[2] * x + e[6] * y + e[10] * z;
		e[15] += e[3] * x + e[7] * y + e[11] * z;
		setElements(e);
		return;
	}

	//SetScale on Model matrix
	void setScale(float x, float y, float z) {
		float e[16];
		for (int n = 0; n < 16; n++){
            e[n] = elements[n];
        }
		e[0] = x;  e[4] = 0;  e[8]  = 0;  e[12] = 0;
		e[1] = 0;  e[5] = y;  e[9]  = 0;  e[13] = 0;
		e[2] = 0;  e[6] = 0;  e[10] = z;  e[14] = 0;
		e[3] = 0;  e[7] = 0;  e[11] = 0;  e[15] = 1;
		setElements(e);
		return;
	};

	//Scale on Model matrix, multiply by x y z
	void scale(float x, float y, float z) {
		float e[16];
		for (int n = 0; n < 16; n++){
            e[n] = elements[n];
        }
		e[0] *= x;  e[4] *= y;  e[8]  *= z;
		e[1] *= x;  e[5] *= y;  e[9]  *= z;
		e[2] *= x;  e[6] *= y;  e[10] *= z;
		e[3] *= x;  e[7] *= y;  e[11] *= z;
		setElements(e);
		return;
	};

	void setRotate( float angle, float x, float y, float z) {
		float s, c, len, rlen, nc, xy, yz, zx, xs, ys, zs;
		float e[16];
		for (int n = 0; n < 16; n++){
	        e[n] = elements[n];
	    }

		angle = M_PI * angle / 180.0;

		s = sin(angle);
		c = cos(angle);

		if (0 != x && 0 == y && 0 == z) {
	    	// Rotation around X axis
			if (x < 0) {
				s = -s;
			}
			e[0] = 1;  e[4] = 0;  e[ 8] = 0;  e[12] = 0;
			e[1] = 0;  e[5] = c;  e[ 9] =-s;  e[13] = 0;
			e[2] = 0;  e[6] = s;  e[10] = c;  e[14] = 0;
			e[3] = 0;  e[7] = 0;  e[11] = 0;  e[15] = 1;
		} else if (0 == x && 0 != y && 0 == z) {
		    // Rotation around Y axis
		    if (y < 0) {
				s = -s;
		    }
		    e[0] = c;  e[4] = 0;  e[ 8] = s;  e[12] = 0;
		    e[1] = 0;  e[5] = 1;  e[ 9] = 0;  e[13] = 0;
		    e[2] =-s;  e[6] = 0;  e[10] = c;  e[14] = 0;
		    e[3] = 0;  e[7] = 0;  e[11] = 0;  e[15] = 1;
		} else if (0 == x && 0 == y && 0 != z) {
	    	// Rotation around Z axis
	    	if (z < 0) {
				s = -s;
			}
		    e[0] = c;  e[4] =-s;  e[ 8] = 0;  e[12] = 0;
		    e[1] = s;  e[5] = c;  e[ 9] = 0;  e[13] = 0;
		    e[2] = 0;  e[6] = 0;  e[10] = 1;  e[14] = 0;
		    e[3] = 0;  e[7] = 0;  e[11] = 0;  e[15] = 1;
		} else {
	    	// Rotation around another axis
	    	len = sqrt(x*x + y*y + z*z);
	    	if (len != 1) {
				rlen = 1 / len;
				x *= rlen;
				y *= rlen;
				z *= rlen;
			}
			nc = 1 - c;
			xy = x * y;
			yz = y * z;
			zx = z * x;
			xs = x * s;
			ys = y * s;
			zs = z * s;

			e[ 0] = x*x*nc +  c;
			e[ 1] = xy *nc + zs;
			e[ 2] = zx *nc - ys;
			e[ 3] = 0;

			e[ 4] = xy *nc - zs;
			e[ 5] = y*y*nc +  c;
			e[ 6] = yz *nc + xs;
			e[ 7] = 0;

			e[ 8] = zx *nc + ys;
			e[ 9] = yz *nc - xs;
			e[10] = z*z*nc +  c;
			e[11] = 0;

			e[12] = 0;
			e[13] = 0;
			e[14] = 0;
			e[15] = 1;
		}
		setElements(e);

		return;
	};

	void rotate( float angle, float x, float y, float z ){
		Matrix4 temp;
		temp.setRotate(angle, x, y, z);
		concat( temp.elements );
	}

	void concat( float (&other)[16] ) {
		int i;
		float ai0, ai1, ai2, ai3;
		float e[16];
		float a[16];
		float b[16];
		// Calculate e = a * b
		for (int n = 0; n < 16; n++){
	        e[n] = elements[n];
	        a[n] = elements[n];
	        b[n] = other[n];
	    }
  
		for (i = 0; i < 4; i++) {
			ai0=a[i];  ai1=a[i+4];  ai2=a[i+8];  ai3=a[i+12];
			e[i]    = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3];
			e[i+4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7];
			e[i+8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11];
			e[i+12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
		}
		setElements(e);

		return;
	};

	// Set Perspective for Perspective Matrix
	void setPerspective(float fovy, float aspect, float near, float far) {
		float rd, s, ct;

		if (near == far || aspect == 0) {
			cerr << 'null frustum' << endl;
		}
		if (near <= 0) {
			cerr << 'near <= 0' << endl;
		}
		if (far <= 0) {
			cerr << 'far <= 0' << endl;
		}

		fovy = M_PI * fovy / 180.0 / 2.0;
  		s = sin(fovy);
		if (s == 0) {
			cerr << 'null frustum' << endl;
		}

		rd = 1.0 / (far - near);
		ct = cos(fovy) / s;

		float e[16]; 
  		for (int n = 0; n < 16; n++){
            e[n] = elements[n];
        }
		e[0]  = ct / aspect;
		e[1]  = 0;
		e[2]  = 0;
		e[3]  = 0;

		e[4]  = 0;
		e[5]  = ct;
		e[6]  = 0;
		e[7]  = 0;

		e[8]  = 0;
		e[9]  = 0;
		e[10] = -(far + near) * rd;
		e[11] = -1;

		e[12] = 0;
		e[13] = 0;
		e[14] = -2 * near * far * rd;
		e[15] = 0;
		setElements(e);
		return;
	}

	//Set Look At for Projection Matrix
	void setLookAt(float eyeX, float eyeY, float eyeZ, 
				float centerX, float centerY, float centerZ, 
				float upX, float upY, float upZ) {
		float fx, fy, fz, rlf, sx, sy, sz, rls, ux, uy, uz;

		fx = centerX - eyeX;
		fy = centerY - eyeY;
		fz = centerZ - eyeZ;

		// Normalize f.
		rlf = 1.0 / sqrt(fx*fx + fy*fy + fz*fz);
		fx *= rlf;
		fy *= rlf;
		fz *= rlf;

		// Calculate cross product of f and up.
		sx = fy * upZ - fz * upY;
		sy = fz * upX - fx * upZ;
		sz = fx * upY - fy * upX;

		// Normalize s.
		rls = 1.0 / sqrt(sx*sx + sy*sy + sz*sz);
		sx *= rls;
		sy *= rls;
		sz *= rls;

		// Calculate cross product of s and f.
		ux = sy * fz - sz * fy;
		uy = sz * fx - sx * fz;
		uz = sx * fy - sy * fx;

		// Set to this.
		float e[16]; 
  		for (int n = 0; n < 16; n++){
            e[n] = elements[n];
        }
		e[0] = sx;
		e[1] = ux;
		e[2] = -fx;
		e[3] = 0;

		e[4] = sy;
		e[5] = uy;
		e[6] = -fy;
		e[7] = 0;

		e[8] = sz;
		e[9] = uz;
		e[10] = -fz;
		e[11] = 0;

		e[12] = 0;
		e[13] = 0;
		e[14] = 0;
		e[15] = 1;
		setElements(e);
		// Translate.
		translate(-eyeX, -eyeY, -eyeZ);
		return;
	};

};

///////////////////////Structs

GLuint vao;

struct uniformStruct {
    GLuint Translation;
    GLuint ViewMatrix;
    GLuint ProjMatrix;
    GLuint ModelMatrix;
    GLuint NormalMatrix;
    GLuint Sampler;
    GLfloat Time;
    float Tx = 0.0;
    float Ty = 0.0;
    float Tz = 0.0;
} u;


struct Primitives {
    GLuint vertexBuffer;
    GLuint normalBuffer;
    GLuint colorBuffer;
    GLuint indexBuffer;
    GLuint texCoordBuffer;
    GLuint textureID;
    int numIndices;
};

Primitives onePlane;
Primitives oneCube;

struct moveMent {
	float px = 0; //Camera Position
	float py = 0;
	float pz = 10;
	float lx = 0; //Look
	float ly = 0; //+1.7
	float lz = 0;
	float turn = -90;
	//Affected by key input
	int moveUp = 0;
	int moveDown = 0;
	int moveLeft = 0;
	int moveRight = 0;

	int delay = 0;
	float ani = 0;
} user; //user parameters


std::string animation = "Still";
float smoothAni = 0;

void NormalKeyHandler(unsigned char key, int x, int y){

	if (key == 32){ //spacebar
		if (animation=="Still"){
			animation = "Running";
		}
		else{
			animation = "Still";
			smoothAni = 0;

		}
	}

	if (key == 122){ // z
		exit(0);
	}
}

void SpecialKeyUpHandler(int key, int x, int y){
	if (key == GLUT_KEY_RIGHT) user.moveRight = 0;
	if (key == GLUT_KEY_LEFT) user.moveLeft = 0;
	if (key == GLUT_KEY_UP) user.moveUp = 0;
	if (key == GLUT_KEY_DOWN) user.moveDown = 0;
}
void SpecialKeyHandler(int key, int x, int y){
	if (key == GLUT_KEY_RIGHT) user.moveRight = 1;
	if (key == GLUT_KEY_LEFT) user.moveLeft = 1;
	if (key == GLUT_KEY_UP) user.moveUp = 1;
	if (key == GLUT_KEY_DOWN) user.moveDown = 1;
}


void smoothNavigate(){
	float e = .15;

	if (user.delay > 0){
		user.delay--;
		return;
	}
	user.delay = 1;

	if (user.moveRight == 1){
		// user.selectx += 1;
	}
	if (user.moveLeft == 1){
		// user.selectx -= 1;
	}
	if (user.moveUp){
		// user.selecty += 1;
		user.py+=1;
		user.ly+=.5;
	}
	if (user.moveDown){
		user.py-=1;
		user.ly-=.5;
		// user.selecty -= 1;
	}

}



GLuint initShader( GLenum type, const char* source ){

    GLuint shader;
    shader = glCreateShader( type );

    ///Compile Vertex shader
    GLint status;
    int length = strlen( source );
    glShaderSource( shader, 1, ( const GLchar ** )&source, &length );
    glCompileShader( shader );
    glGetShaderiv( shader, GL_COMPILE_STATUS, &status );

    if( status == GL_FALSE )
    {
        fprintf( stderr, "Fragment shader compilation failed.\n" );

        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        std::vector<GLchar> errorLog(maxLength);
        glGetShaderInfoLog(shader,  maxLength, &maxLength, &errorLog[0]);

        for(int n = 0; n < maxLength; n++){
            cerr << errorLog[n];
        }
        cerr << endl;

        glDeleteShader( shader ); // Don't leak the shader.
        return -1;
    }   
    return shader;
}


void initCube(Primitives &o, const char* file, float r, float g, float b) {
	// Create a cube
	//    v6----- v5
	//   /|      /|
	//  v1------v0|
	//  | |     | |
	//  | |v7---|-|v4
	//  |/      |/
	//  v2------v3

	const GLfloat vertices[] = {
	    /*X,  Y,  Z  */
		 .5, .5, .5,  -.5, .5, .5,  -.5,-.5, .5,   .5,-.5, .5,  
		 .5, .5, .5,   .5,-.5, .5,   .5,-.5,-.5,   .5, .5,-.5,
		 .5, .5, .5,   .5, .5,-.5,  -.5, .5,-.5,  -.5, .5, .5,
		-.5, .5, .5,  -.5, .5,-.5,  -.5,-.5,-.5,  -.5,-.5, .5, 
		-.5,-.5,-.5,   .5,-.5,-.5,   .5,-.5, .5,  -.5,-.5, .5, 
		 .5,-.5,-.5,  -.5,-.5,-.5,  -.5, .5,-.5,   .5, .5,-.5 
    };

    float n = 3;
    const GLfloat texCoords[] = {
		1.0, 1.0,   0.0, 1.0,   0.0, 0.0,   1.0, 0.0,    // v0-v1-v2-v3 front
		0.0, 1.0,   0.0, 0.0,   1.0, 0.0,   1.0, 1.0,    // v0-v3-v4-v5 right
		n,   0.0,   n,n,        0.0, n,     0.0, 0.0,    // v0-v5-v6-v1 up
		1.0, 1.0,   0.0, 1.0,   0.0, 0.0,   1.0, 0.0,    // v1-v6-v7-v2 left
		0.0, 0.0,   1.0, 0.0,   1.0, 1.0,   0.0, 1.0,    // v7-v4-v3-v2 down
		0.0, 0.0,   1.0, 0.0,   1.0, 1.0,   0.0, 1.0     // v4-v7-v6-v5 back
    };

    /*const GLfloat colors[] = {
        R,   G,   B,  A,
	    0.1, 0.1, 1.0,1,  0.1, 0.1, 1.0,1,  0.1, 0.1, 1.0,1,  0.1, 0.1, 1.0,1,
	    0.1, 1.0, 0.1,1,  0.1, 1.0, 0.1,1,  0.1, 1.0, 0.1,1,  0.1, 1.0, 0.1,1,
	    1.0, 0.1, 0.1,1,  1.0, 0.1, 0.1,1,  1.0, 0.1, 0.1,1,  1.0, 0.1, 0.1,1,
	    1.0, 1.0, 0.1,1,  1.0, 1.0, 0.1,1,  1.0, 1.0, 0.1,1,  1.0, 1.0, 0.1,1,
	    1.0, 1.0, 1.0,1,  1.0, 1.0, 1.0,1,  1.0, 1.0, 1.0,1,  1.0, 1.0, 1.0,1,
	    0.1, 1.0, 1.0,1,  0.1, 1.0, 1.0,1,  0.1, 1.0, 1.0,1,  0.1, 1.0, 1.0,1
    }; */

    const GLfloat colors[] = {
	    r, g, b,1,  r, g, b,1,  r, g, b,1,  r, g, b,1,
	    r, g, b,1,  r, g, b,1,  r, g, b,1,  r, g, b,1,
	    r, g, b,1,  r, g, b,1,  r, g, b,1,  r, g, b,1,
	    r, g, b,1,  r, g, b,1,  r, g, b,1,  r, g, b,1,
	    r, g, b,1,  r, g, b,1,  r, g, b,1,  r, g, b,1,
	    r, g, b,1,  r, g, b,1,  r, g, b,1,  r, g, b,1,
    };

	const GLfloat normals[] {
		0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0,  // v0-v1-v2-v3 front
		1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,  // v0-v3-v4-v5 right
		0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,  // v0-v5-v6-v1 up
	   -1.0, 0.0, 0.0,  -1.0, 0.0, 0.0,  -1.0, 0.0, 0.0,  -1.0, 0.0, 0.0,  // v1-v6-v7-v2 left
		0.0,-1.0, 0.0,   0.0,-1.0, 0.0,   0.0,-1.0, 0.0,   0.0,-1.0, 0.0,  // v7-v4-v3-v2 down
		0.0, 0.0,-1.0,   0.0, 0.0,-1.0,   0.0, 0.0,-1.0,   0.0, 0.0,-1.0   // v4-v7-v6-v5 back
	};

    const GLuint indices[] = {
		 0, 1, 2,   0, 2, 3,    // front
		 4, 5, 6,   4, 6, 7,    // right
		 8, 9,10,   8,10,11,    // up
		12,13,14,  12,14,15,    // left
		16,17,18,  16,18,19,    // down
		20,21,22,  20,22,23     // back    	
    };
  
    o.numIndices = sizeof(indices) / 4;

    // Create buffer objects
	glGenBuffers( 1, &o.vertexBuffer );
	glGenBuffers( 1, &o.colorBuffer );
	glGenBuffers( 1, &o.indexBuffer );
	glGenBuffers( 1, &o.texCoordBuffer );
	glGenTextures( 1, &o.textureID );

    //Position
    glBindBuffer( GL_ARRAY_BUFFER, o.vertexBuffer);
    glBufferData( GL_ARRAY_BUFFER, sizeof(vertices), &vertices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(a_Position, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(a_Position);

    //Normal
    glBindBuffer( GL_ARRAY_BUFFER, o.normalBuffer);
    glBufferData( GL_ARRAY_BUFFER, sizeof(normals), &normals[0], GL_STATIC_DRAW);
    glVertexAttribPointer(a_Normal, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(a_Normal);

    //Color
    glBindBuffer( GL_ARRAY_BUFFER, o.colorBuffer);
    glBufferData( GL_ARRAY_BUFFER, sizeof(colors), &colors[0], GL_STATIC_DRAW);
    glVertexAttribPointer(a_Color, 4, GL_FLOAT, GL_FALSE, 0, 0 );
    glEnableVertexAttribArray(a_Color);

    //Texture Coordinates
    glBindBuffer( GL_ARRAY_BUFFER, o.texCoordBuffer);
    glBufferData( GL_ARRAY_BUFFER, sizeof(texCoords), &texCoords[0], GL_STATIC_DRAW);
    glVertexAttribPointer(a_TexCoord, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(a_TexCoord);

    //Bind Texture
    glActiveTexture( GL_TEXTURE0);
    glBindTexture(   GL_TEXTURE_2D, o.textureID);

    if (file != "None"){
	    int w, h;
	    unsigned char* image = SOIL_load_image(file, &w, &h, 0, SOIL_LOAD_RGB);
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w,h,
	    	0, GL_RGB, GL_UNSIGNED_BYTE, image);
	    SOIL_free_image_data(image);
	}else{
	    float pixels[] = {
	    	1.0f, 1.0f, 1.0f,   1.0f, 1.0f, 1.0f,
	    	1.0f, 1.0f, 1.0f,   1.0f, 1.0f, 1.0f,
	    	// 1.0f, 1.0f, 1.0f,   0.0f, 0.0f, 0.0f,
	    	// 0.0f, 0.0f, 0.0f,   1.0f, 1.0f, 1.0f,
	    };
	    //Target active unit, level, internalformat, width, height, border, format, type, data
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2,
	    	0, GL_RGB, GL_FLOAT, pixels);
    }

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenerateMipmap(GL_TEXTURE_2D);

    //Index Buffer
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, o.indexBuffer);
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), &indices[0], GL_STATIC_DRAW);

    //No Buffer Bound
	glBindBuffer( GL_ARRAY_BUFFER, NULL);
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, NULL);

}


void initPlane(Primitives &o, const char* file){
	// Create a Plane
	//  v1------v0
	//  |       | 
	//  |       |
	//  |       |
	//  v2------v3

    const GLfloat vertices[] = {
     //  X   Y  Z
		.5, .5,  0,  -.5, .5,  0,  -.5,-.5,  0,   .5,-.5,  0,  
    };

    const GLfloat colors[] = {
     // R, G, B, A,
        0, 1, 1, 1,
        1, 1, 0, 1, 
        1, 0, 1, 1,

        1, 0, 1, 1, 
        0, 1, 1, 1, 
        1, 1, 0, 1,
    };

    //Coordinates for the 4 corners of image samples
    float n = 1;
    const GLfloat texCoords[] = {
		n, 0, 0,0,  0,n,  n, n, // v0-v5-v6-v1 up
    };

    //3-2
    //|x|
    //0-1
    const GLuint indices[] = {
		 0, 1, 2,   0, 2, 3, // front 
    };

    o.numIndices = sizeof(indices) / 4;

    // Create buffer objects (VBO's)
	glGenBuffers( 1, &o.vertexBuffer);
	glGenBuffers( 1, &o.normalBuffer);
	glGenBuffers( 1, &o.colorBuffer );
	glGenBuffers( 1, &o.indexBuffer );
	glGenBuffers( 1, &o.texCoordBuffer );
	glGenTextures( 1, &o.textureID );

    //Position
    glBindBuffer( GL_ARRAY_BUFFER, o.vertexBuffer);
    glBufferData( GL_ARRAY_BUFFER, sizeof(vertices), &vertices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(a_Position, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(a_Position);

    //Color
    glBindBuffer( GL_ARRAY_BUFFER, o.colorBuffer);
    glBufferData( GL_ARRAY_BUFFER, sizeof(colors), &colors[0], GL_STATIC_DRAW);
    glVertexAttribPointer(a_Color, 4, GL_FLOAT, GL_FALSE, 0, 0 );
    glEnableVertexAttribArray(a_Color);

    //Index Buffer
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, o.indexBuffer);
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), &indices[0], GL_STATIC_DRAW);

	//Texture Coordinates
    glBindBuffer( GL_ARRAY_BUFFER, o.texCoordBuffer);
    glBufferData( GL_ARRAY_BUFFER, sizeof(texCoords), &texCoords[0], GL_STATIC_DRAW);
    glVertexAttribPointer(a_TexCoord, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(a_TexCoord);

    //Bind Texture
    glEnable(GL_TEXTURE_2D);
    glActiveTexture( GL_TEXTURE0);
    glBindTexture(   GL_TEXTURE_2D, o.textureID);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (file != "None"){
	    int w, h;
	    unsigned char* image = SOIL_load_image(file, &w, &h, 0, SOIL_LOAD_RGB);
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w,h,
	    	0, GL_RGB, GL_UNSIGNED_BYTE, image);
        SOIL_free_image_data(image);
	}else{	
		int c = 16;
   		float pixels[c*c*3];
   		for (int i = 0; i < c*c*3; i++){
   			pixels[i] = 1.0f;
   		}
	    //Target active unit, level, internalformat, width, height, border, format, type, data
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, c, c,
	    	0, GL_RGB, GL_FLOAT, pixels);
    }

    //Set various texture sample settings
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenerateMipmap(GL_TEXTURE_2D);

    //Set No Buffer Bound
	glBindBuffer( GL_ARRAY_BUFFER, NULL);
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, NULL);
}



Matrix4 viewMatrix;
Matrix4 projMatrix;
Matrix4 modelMatrix;
Matrix4 normalMatrix;

void render(Primitives &o){

	//Activate Vertex Coordinates
    glBindBuffer( GL_ARRAY_BUFFER, o.vertexBuffer);
    glVertexAttribPointer(a_Position, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(a_Position);

	//Normals
    glBindBuffer( GL_ARRAY_BUFFER, o.normalBuffer);
    glVertexAttribPointer(a_Normal, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(a_Normal);

  	//Activate Color Coordinates
	glBindBuffer( GL_ARRAY_BUFFER, o.colorBuffer);
	glVertexAttribPointer(a_Color, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(a_Color);

	//Activate Texture Coordinates
	glBindBuffer( GL_ARRAY_BUFFER, o.texCoordBuffer);
	glVertexAttribPointer(a_TexCoord, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(a_TexCoord);

	//Bind Texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture( GL_TEXTURE_2D, o.textureID);

	//Bind Indices
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, o.indexBuffer);

	//Normal Matrix
	normalMatrix.copyFrom(modelMatrix);
	normalMatrix.transpose();

	//Update uniforms in frag vertex  //1 denotes number of matrixes to update
    glUniformMatrix4fv( u.ViewMatrix, 1, GL_TRUE, viewMatrix.elements);
    glUniformMatrix4fv( u.ProjMatrix, 1, GL_TRUE, projMatrix.elements);
    glUniformMatrix4fv( u.ModelMatrix, 1, GL_TRUE, modelMatrix.elements);
    glUniformMatrix4fv( u.NormalMatrix, 1, GL_TRUE, normalMatrix.elements);
    glUniform1i( u.Sampler, 0);

    //After Binding all new attributes and updating Uniforms...
    //DrawElements allows to display Cube, etc, with fewer indices
    glDrawElements( GL_TRIANGLES, o.numIndices, GL_UNSIGNED_INT, 0);
   
}

struct Body {

	std::string name;
	Primitives part;
	Matrix4 modelMatrix;
	Matrix4 modelToPass;
	struct Body* next[5];

	Body(){
		for(int n = 0; n < 5; n++){
			next[n] = NULL;
			modelMatrix.setIdentity();
			modelToPass.setIdentity();
		}
	}

} Body;


float sm(float mult){
	return smoothAni * mult; 
}

void RunningAni(struct Body* chara, Matrix4& pass){
	if (chara->name == "Torso"){

		user.ani += .115;
		if (smoothAni < 1)
			smoothAni += .01;

		pass.translate(0, sm(  sin(user.ani*2-4.7)*.1  ), 0);
		pass.rotate( sin(user.ani)*sm(4) , 0,1,0);
	}
	if (chara->name == "Head"){
		pass.rotate( -sin(user.ani)*sm(4), 0,1,0);
	}
	//Arms
	if (chara->name == "ArmRightUpper"){
		pass.rotate( sin(user.ani)*sm(3), 1,0,0);
	}
	if (chara->name == "ArmLeftUpper"){
		pass.rotate( -sin(user.ani)*sm(3), 1,0,0);
	}
	if (chara->name == "ArmRightLower"){
		pass.rotate( (sin(user.ani)-1)*sm(2), 1,0,0);
		pass.translate(0,0,sm(.05));
	}
	if (chara->name == "ArmLeftLower"){
		pass.rotate( (-sin(user.ani)-1)*sm(2), 1,0,0);
		pass.translate(0,0,sm(.05));
	}
	//Lower body
	if (chara->name == "Pelvis"){
		pass.rotate( -sin(user.ani)*sm(8), 0,1,0);
	}
	if (chara->name == "LegsRightUpper"){
		//Align straight from pelvis...
		pass.rotate( sin(user.ani)*sm(4), 0,1,0);
		//Tilted sin graph!
		pass.rotate( -sin(user.ani + sin(user.ani)/2 )*sm(8), 1,0,0);
	}
	if (chara->name == "LegsLeftUpper"){
		pass.rotate( sin(user.ani)*sm(4), 0,1,0);
		//Tilted sin graph!!
		pass.rotate( sin(user.ani - sin(user.ani)/2 )*sm(8), 1,0,0);
	}
	if (chara->name == "LegsRightLower"){
		pass.translate(0,0,sm(.2));
		pass.rotate( -(sin(user.ani + sin(user.ani)/2 )-1)*sm(4), 1,0,0);		
	}
	if (chara->name == "LegsLeftLower"){
		pass.translate(0,0,sm(.2));
		pass.rotate( (sin(user.ani - sin(user.ani)/2 )+1)*sm(4), 1,0,0);		
	}

}


void recurBody(struct Body* chara, Matrix4 pass){

	if (animation == "Running"){
		RunningAni(chara, pass);
	}

	//Copy matrix of parent part
	modelMatrix.copyFrom(pass);
	modelMatrix.concat(chara->modelToPass.elements);

	//Temp matrix for transformations that will not pass
	Matrix4 tempMatrix;
	tempMatrix.copyFrom(modelMatrix);

	modelMatrix.concat(chara->modelMatrix.elements);

	render(chara->part);

	//Loop through any possible body parts and send transformations
	for(int n = 0; n < 5; n++){

		if (chara->next[n] == NULL) continue;
		recurBody(chara->next[n], tempMatrix);

	}
}

void display(int te){

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutTimerFunc(1000.0/60.0, display, 1);
    smoothNavigate(); //Update user movement

    u.Tx += .01;
    glUniform4f(u.Translation, u.Tx, u.Ty, u.Tz, 0.0);

    //Initialize Matrices
	projMatrix.setPerspective(90.0, (float)WIDTH/(float)HEIGHT, 0.1, 1450.0);
	//eyeX, eyeY, eyeZ, (at)centerX, (at)centerY, (at)centerZ, upX, upY, upZ
	// viewMatrix.setLookAt(cos(u.Tx)*10, user.py+5, sin(u.Tx)*10,    user.lx, user.ly, user.lz,    0.0, 1.0, 0.0);
	viewMatrix.setLookAt(5, user.py+5, 5,    user.lx, user.ly, user.lz,    0.0, 1.0, 0.0);


	//Update Time
	glUniform1f( u.Time, u.Tx );

	// modelMatrix.setScale(5,5,5);
	// modelMatrix.rotate(u.Tx, 0,0,1);
	// modelMatrix.translate(0.0,0,-u.Tx*.1);
	// render(onePlane);

	struct Body* chara = &Body;

	modelMatrix.setIdentity();
	recurBody(chara, modelMatrix);
   
    glutSwapBuffers();
}




int main(int argc, char** argv)
{
	srand(time(0));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitContextVersion (3, 2);
    glutInitContextFlags (GLUT_FORWARD_COMPATIBLE | GLUT_DEBUG);
    
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("OpenGL - Character");
    glutSpecialFunc(SpecialKeyHandler);
    glutSpecialUpFunc(SpecialKeyUpHandler);
    glutKeyboardFunc(NormalKeyHandler);

    // Initialize GLEW
    glewExperimental = GL_TRUE; 
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    if(GLEW_VERSION_3_0)
    {
        //cerr << "GlEW Available";
    }else
        return 0;

    GLuint vs, fs, program;
    vs = initShader( GL_VERTEX_SHADER, vertex_source );
    fs = initShader( GL_FRAGMENT_SHADER, fragment_source );
    if (vs == -1 || fs == -1){ return 0; }

    //Create and use shader program
    program = glCreateProgram();
    glAttachShader( program, vs );
    glAttachShader( program, fs );

    //Must link after BindAttrib
    glLinkProgram( program );
    glUseProgram( program );

	glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LESS );

    //Clear Color
	glClearColor(0,0,0,  1.0);

    glViewport( 0, 0, WIDTH, HEIGHT );

    //VAO?
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );

    //Storage Locations for Uniforms
    u.Translation = glGetUniformLocation( program, "u_Translation" );
	u.ProjMatrix = glGetUniformLocation( program, "u_ProjMatrix");
	u.ViewMatrix = glGetUniformLocation( program, "u_ViewMatrix");
	u.ModelMatrix = glGetUniformLocation( program, "u_ModelMatrix");
	u.NormalMatrix = glGetUniformLocation( program, "u_NormalMatrix");
	u.Sampler = glGetUniformLocation( program, "u_Sampler");
	u.Time = glGetUniformLocation( program, "u_Time");

    //Storage locations for Attributes (like a local reference to the shader variable)
    glBindAttribLocation( program, a_Position, "a_Position" );
   	glBindAttribLocation( program, a_Normal, "a_Normal" );
    glBindAttribLocation( program, a_Color, "a_Color" );
    glBindAttribLocation( program, a_TexCoord, "a_TexCoord" );

    //Buffers (a_ attributes)
    initPlane(onePlane, "None");
    initCube(oneCube, "None", 1,1,1);

    //Create Body of Character
    initCube(Body.part, "None", 0,0,.54);
    Body.modelMatrix.setScale(2,3,1); //2,3,1
    Body.name = "Torso";

    //Add Head to Body
    struct Body Head; Head.name = "Head";
    initCube(Head.part, "None", .55,.33,.14);
    Head.modelMatrix.setScale(1,1.4,1);
    Head.modelToPass.setTranslate(0,2.4,0);
    Body.next[3] = &Head;

    //Add Arm Upper to Body
    struct Body ArmRightUpper; ArmRightUpper.name = "ArmRightUpper";
    struct Body ArmLeftUpper; ArmLeftUpper.name = "ArmLeftUpper";
    initCube(ArmRightUpper.part, "None", 0,0,.54);
    initCube(ArmLeftUpper.part, "None", 0,0,.54);

    ArmRightUpper.modelMatrix.setScale(.7,2,.7);
    ArmLeftUpper.modelMatrix.setScale(.7,2,.7);
    ArmRightUpper.modelToPass.setTranslate(1.6,.5,0);
    ArmLeftUpper.modelToPass.setTranslate(-1.6,.5,0);
    Body.next[0] = &ArmRightUpper;
    Body.next[1] = &ArmLeftUpper;

    //Add Arm Lower to Arm Upper
    struct Body ArmRightLower; ArmRightLower.name = "ArmRightLower";
    struct Body ArmLeftLower; ArmLeftLower.name = "ArmLeftLower";
    initCube(ArmRightLower.part, "None", 1,1,1);
    initCube(ArmLeftLower.part, "None", 1,1,1);

    ArmRightLower.modelMatrix.setScale(.6,2,.6);
    ArmLeftLower.modelMatrix.setScale(.6,2,.6);
    ArmRightLower.modelToPass.setTranslate(0,-2.1,0);
    ArmLeftLower.modelToPass.setTranslate(0,-2.1,0);
    Body.next[0]->next[0] = &ArmRightLower;
    Body.next[1]->next[0] = &ArmLeftLower;

    //Add Hands to Arms
    struct Body HandRight; HandRight.name = "HandRight";
    struct Body HandLeft; HandLeft.name = "HandLeft";
    initCube(HandRight.part, "None", .55,.33,.14);
    initCube(HandLeft.part, "None", .55,.33,.14);

	HandRight.modelMatrix.setScale(.6,1.2,.8);
    HandLeft.modelMatrix.setScale(.6,1.2,.8);
    HandRight.modelToPass.setTranslate(0,-1.8,0);
    HandLeft.modelToPass.setTranslate(0,-1.8,0);
    Body.next[0]->next[0]->next[0] = &HandRight;
    Body.next[1]->next[0]->next[0] = &HandLeft;

    //Add Pelvis to Body
    struct Body Pelvis; Pelvis.name = "Pelvis";
    initCube(Pelvis.part, "None", 1,1,1);
    Pelvis.modelToPass.setTranslate(0,-2.5,0);
    Pelvis.modelMatrix.setScale(2,1.5,1);
    Body.next[2] = &Pelvis;

    //Add Legs Upper to Pelvis
    struct Body LegsRightUpper; LegsRightUpper.name = "LegsRightUpper";
    struct Body LegsLeftUpper; LegsLeftUpper.name = "LegsLeftUpper";
    initCube(LegsRightUpper.part, "None", 1,1,1);
    initCube(LegsLeftUpper.part, "None", 1,1,1);

    LegsRightUpper.modelMatrix.setScale(1,3,1);
    LegsLeftUpper.modelMatrix.setScale(1,3,1);
    LegsRightUpper.modelMatrix.rotate(45,0,1,0);
    LegsLeftUpper.modelMatrix.rotate(-45,0,1,0);
    LegsRightUpper.modelToPass.setTranslate(.7,-2.5,0);
    LegsLeftUpper.modelToPass.setTranslate(-.7,-2.5,0);
    Body.next[2]->next[0] = &LegsRightUpper;
    Body.next[2]->next[1] = &LegsLeftUpper;

    //Add Legs Lower to Legs Upper
    struct Body LegsRightLower; LegsRightLower.name = "LegsRightLower";
    struct Body LegsLeftLower; LegsLeftLower.name = "LegsLeftLower";
    initCube(LegsRightLower.part, "None", 1,1,1);
    initCube(LegsLeftLower.part, "None", 1,1,1);

    LegsRightLower.modelMatrix.setScale(.8,3,.8);
    LegsLeftLower.modelMatrix.setScale(.8,3,.8);
    LegsRightLower.modelMatrix.rotate(45,0,1,0);
    LegsLeftLower.modelMatrix.rotate(-45,0,1,0);
    LegsRightLower.modelToPass.setTranslate(0,-3,0);
    LegsLeftLower.modelToPass.setTranslate(0,-3,0);
    Body.next[2]->next[0]->next[0] = &LegsRightLower;
    Body.next[2]->next[1]->next[0] = &LegsLeftLower;

    //Add Feet to Legs Lower
    struct Body FootRight; FootRight.name = "FootRight";
    struct Body FootLeft; FootLeft.name = "FootLeft";
    initCube(FootRight.part, "None", .5,.5,.5);
    initCube(FootLeft.part, "None", .5,.5,.5);

    FootRight.modelMatrix.setScale(.8,.7, 1.3);
    FootLeft.modelMatrix.setScale( .8,.7, 1.3);
    FootRight.modelToPass.setTranslate(0,-2,.25);
    FootLeft.modelToPass.setTranslate(0,-2,.25);
    Body.next[2]->next[0]->next[0]->next[0] = &FootRight;
    Body.next[2]->next[1]->next[0]->next[0] = &FootLeft;


    //Set Frames Per Second
    glutTimerFunc(1000.0/60.0, display, 1);
    glutMainLoop();

    return 0;
}
