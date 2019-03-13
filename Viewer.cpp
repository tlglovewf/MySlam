//
//  Viewer.cpp
//  MySlam
//
//  Created by TuLigen on 2019/3/12.
//

#include "Viewer.h"
#include <pangolin/pangolin.h>
#include <thread>
void Viewer::test()
{
    pangolin::CreateWindowAndBind("Main",800,800);
    
    glEnable(GL_DEPTH_TEST);
    
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
                                      pangolin::ProjectionMatrix(800,800,400,400,400,400,0.2,100),
                                      pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
                                      );
    
    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -800.0f/800.0f)
    .SetHandler(&handler);
    
    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glClearColor(1.0, 1.0, 1.0, 1.0);
        
        d_cam.Activate(s_cam);
        
        // Render OpenGL Cube
        
//        pangolin::glDrawColouredCube();
        pangolin::glDrawAxis(5);
        
        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
}
