%/*************************************************************************
%* File Name     : fem_test.m
%* Code Title    : 
%* Programmer    :
%* Affiliation   : 
%* Creation Date : 2020/10/11
%* Language      : Matlab
%* Version       : 1.0.0
%**************************************************************************
%* NOTE:
%*  
%*************************************************************************/

clear all; close all;  %‘S‚Ä‚Ì•Ï”‚ÆFigure‚ğíœ

%/*************************************************************************

model = createpde('structural','static-solid');

importGeometry(model, 'simulation_naca0015.stl');

figure(1);
pdegplot(model,"FaceLabels","on")
view(30,30);
title("NACA0015 morphing")

structuralProperties(model,"YoungsModulus",0.003e9,"PoissonsRatio",0.49);
structuralBC(model,"Face",10,"Constraint","fixed");
structuralBoundaryLoad(model,"Face",107,"Pressure",60);

generateMesh(model);
figure(2)
pdeplot3D(model)

result = solve(model);

figure(3)
pdeplot3D(model,'ColorMapData',result.Displacement.uy)
title('y-displacement')
colormap('jet')