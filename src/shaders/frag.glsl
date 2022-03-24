const float PI = 3.1415926535897932384626433832795;
const float TAU = 2.* PI;
uniform vec3 uColor;
uniform vec3 uPosition;
uniform vec3 uRotation;
uniform vec2 uResolution;
// uniform sampler2D uTexture;
// uniform sampler2D uVideo;
// uniform sampler2D uVideo2;
uniform vec2 uMouse;
uniform float uValueA;
uniform float uValueB;
uniform float uValueC;
uniform float uValueD;


varying float vDistort;
varying vec2 vUv;
varying float vElevation;
varying float vTime;
varying vec3 vNorm;

precision highp float;

#define PI 3.14159265359

vec2 brownConradyDistortion(in vec2 uv, in float k1, in float k2)
{
    uv = uv * 2.0 - 1.0;	// brown conrady takes [-1:1]

    // positive values of K1 give barrel distortion, negative give pincushion
    float r2 = uv.x*uv.x + uv.y*uv.y;
    uv *= 1.0 + k1 * r2 + k2 * r2 * r2;

    // tangential distortion (due to off center lens elements)
    // is not modeled in this function, but if it was, the terms would go here

    uv = (uv * .5 + .5);	// restore -> [0:1]
    return uv;
}

float smoothIntersectSDF(float distA, float distB, float k )
{
  float h = clamp(0.5 - 0.5*(distA-distB)/k, 0., 1.);
  return mix(distA, distB, h ) + k*h*(1.-h);
}

float smoothUnionSDF(float distA, float distB, float k ) {
  float h = clamp(0.5 + 0.5*(distA-distB)/k, 0., 1.);
  return mix(distA, distB, h) - k*h*(1.-h);
}

// vec4 smoothDifferenceSDF(vec4 a, vec4 b, float k)
// {
//   float h = clamp(0.5 - 0.5*(a.w+b.w)/k, 0., 1.);
//   vec3 c = mix(a.rgb,b.rgb,h);
//   float d = mix(a.w, -b.w, h ) + k*h*(1.-h);
//
//   return vec4(c,d);
// }

vec3 opTwist( in vec3 p )
{
    float k = sin(vTime) ; // or some other amount
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y);
    return q;
}


float smoothDifferenceSDF(float distA, float distB, float k) {
  float h = clamp(0.5 - 0.5*(distB+distA)/k, 0., 1.);
  return mix(distA, -distB, h ) + k*h*(1.-h);
}

// vec4 smoothUnionSDF(vec4 a, vec4 b, float k )
// {
//   float h = clamp(0.5 + 0.5*(a.w-b.w)/k, 0., 1.);
//   vec3 c = mix(a.rgb,b.rgb,h);
//   float d = mix(a.w, b.w, h) - k*h*(1.-h);
//
//   return vec4(c,d);
// }

vec4 sdSphere(vec3 p, float r, vec3 offset, vec3 col )
{
  float d = length(p - offset) - r;
  return vec4(d, col);
}

vec4 sdBox(vec3 p, vec3 c, vec3 col) {
  vec3 q = abs(p) - c;
  return vec4(length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) , col);
}



vec4 sdFloor(vec3 p, vec3 col) {
  float d = p.y + 1.;
  return vec4(d, col);
}

vec4 minWithColor(vec4 obj1, vec4 obj2) {
  if (obj2.x < obj1.x) return obj2; // The x component of the object holds the "signed distance" value
  return obj1;
}

// vec4 smoothDifferenceSDF(vec4 a, vec4 b, float k)
// {
//   float h = clamp(0.5 - 0.5*(a.w+b.w)/k, 0., 1.);
//   vec3 c = mix(a.rgb,b.rgb,h);
//   float d = mix(a.w, -b.w, h ) + k*h*(1.-h);
//
//   return vec4(c,d);
// }

mat2 rot (float a) {
	return mat2(cos(a),sin(a),-sin(a),cos(a));
}


vec4 intersectSDF(vec4 a, vec4 b) {
    return a.w > b.w ? a : b;
}

vec4 unionSDF(vec4 a, vec4 b) {
    return a.w < b.w? a : b;
}

vec4 differenceSDF(vec4 a, vec4 b) {
    return a.w > -b.w? a : vec4(b.rgb,-b.w);
}

// float opRep( in vec3 p, in vec3 c, in sdf3d primitive )
// {
//     vec3 q = mod(p+0.5*c,c)-0.5*c;
//     return primitive( q );
// }
vec3 opRep( vec3 p, vec3 c ) {
  float displacement = sin(9. * p.x + vTime * .5) * sin(3. * p.y + vTime) * sin(3. * p.z + vTime) * 0.25 ;
    vec3 q = mod(p,c)-0.5*c;
    q.y +=vUv.x * 5.;
    q.y -=2.;
    return q * 1.;
}

vec3 opCheapBend(  in vec3 p )
{
    float k = 10. * sin(vTime); // or some other amount
    float c = cos(k*p.x);
    float s = sin(k*p.x);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xy,p.z);
    return q;
}

vec4 sdLink( vec3 p, float le, float r1, float r2, vec3 color )
{
  vec3 q = vec3( p.x, max(abs(p.y)-le,0.0), p.z );
  return vec4(length(vec2(length(q.xy)-r1,q.z)) - r2, color);
}

vec4 sdTorus( vec3 p, vec2 t , vec3 color)
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return vec4(length(q)-t.y, color);
}
void coswarp(inout vec3 trip, float warpsScale ){

  trip.xyz += warpsScale * .1 * cos(3. * trip.yzx + (vTime * .25));
  trip.xyz += warpsScale * .05 * cos(11. * trip.yzx + (vTime * .25));
  trip.xyz += warpsScale * .025 * cos(17. * trip.yzx + (vTime * .25));
  // trip.xyz += warpsScale * .0125 * cos(21. * trip.yzx + (vTime * .25));
}
//
// GLSL textureless classic 3D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-10-11
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/stegu/webgl-noise
//

vec3 mod289(vec3 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x)
{
  return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade(vec3 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise(vec3 P)
{
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}


vec4 sdScene(vec3 p) {
  // vec3 p2 = p;
    vec3 p3 = p;
      vec3 p4 = p;


  float warpsScale = 3.;
  vec3 color1 = vec3(1. * uValueA, vUv.y, vUv.x);

  color1.xyz += warpsScale * .1 * cos(3. * color1.yzx + vTime);
  color1.xyz += warpsScale * .05 * cos(11. * color1.yzx + vTime);
  color1.xyz += warpsScale * .025 * cos(17. * color1.yzx + vTime);
  color1.xyz += warpsScale * .0125 * cos(21. * color1.yzx + vTime);





  vec3 color3 = vec3(uValueB + uValueC, vUv.x,vUv.y);
  color3.xyz += warpsScale * .1 * sin(3. * color3.yzx + vTime);
  color3.xyz += warpsScale * .05 * cos(11. * color3.yzx + vTime);
  color3.xyz += warpsScale * .025 * cos(17. * color3.yzx + vTime);
  color3.xyz += warpsScale * .0125 * cos(21. * color3.yzx + vTime);

  vec3 color4 = vec3(vUv.y, vUv.x, 0.);
  color4.xyz += warpsScale * .1 * sin(3. * color4.yzx + vTime);
  color4.xyz += warpsScale * .05 * cos(11. * color4.yzx + vTime);
  color4.xyz += warpsScale * .025 * cos(17. * color4.yzx + vTime);
  color4.xyz += warpsScale * .0125 * cos(21. * color4.yzx + vTime);



  float displacement = sin(3. * p.x + vTime * .5 ) * sin(3. * p.y + vTime  ) * sin(3. * p.z + vTime) * 0.25 ;

  p3.x -1.5;
  p4.x -1.5;


  p3 = opRep(p3, vec3(6. ,0 , 0.));
  p4 = opRep(p4, vec3(6. ,0, 0.));
  // p3 = opCheapBend(p3);

  // p3 += cnoise(p3 * 1.);


  // p3 = opTwist(p3);

 p4.xz *= rot(vTime * .5);
 // p3.xz *= rot(vTime * .5);
 //
 // p3.xy *= rot(vTime * .5) * -1.;
 // p3.xz *= rot(vTime * .5) * -1.;

 // p.xy *= rot(vTime * .5 * -1.);

  vec4 sphereLeft = sdSphere(p3 + displacement  ,1. , vec3(0., 0, .0), color1 );



  vec4 torus = sdTorus(p4, vec2(1.7 , .1 ), color3);

  // vec4 hmm = smoothUnionSDF(link, link2, .005);
  // float blah = mix(link, link2, sin(vTime * .5) * .5 +1.);

   vec4 mixed = minWithColor(sphereLeft, torus);
   // mixed = minWithColor(mixed, cube3);
  return  mixed;
}

const int MAX_MARCHING_STEPS = 50;
const float MIN_DIST = 0.0;
const float MAX_DIST = 250.0;
const float PRECISION = 0.001;


vec4 rayMarch(vec3 ro, vec3 rd, float start, float end) {
  float depth = start;
  vec4 co; // closest object

  for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
    vec3 p = ro + depth * rd;
    co = sdScene(p);
    depth += co.x;
    if (co.x < PRECISION || depth > end) break;
  }

  vec3 col = vec3(co.yzw);

  return vec4(depth, col);
}

vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(1.0, -1.0) * 0.0005; // epsilon
    return normalize(
      e.xyy * sdScene(p + e.xyy).x +
      e.yyx * sdScene(p + e.yyx).x +
      e.yxy * sdScene(p + e.yxy).x +
      e.xxx * sdScene(p + e.xxx).x);
}

float Star(vec2 uv, float flare) {
	float d = length(uv);
    float m = .05/d;

    float rays = max(0., 1.-abs(uv.x*uv.y*1000.));
    m += rays*flare;
    uv *= rot(3.1415/4.);
    rays = max(0., 1.-abs(uv.x*uv.y*1000.));
    m += rays*.3*flare;

    m *= smoothstep(1., .2, d);
    return m;
}

void main( )
{
  vec2 uv = (gl_FragCoord.xy - uResolution * .5) / uResolution.yy;
  vec2 uv2 = uv * (24.0 * uValueA) *  cnoise(vec3(uv *rot(vTime * .1), 1.) * (40. * uValueB)) ;      // Scale up the space by 3
   uv2 = fract(uv2 );
  vec3 backgroundColor = vec3(uv2.x, uv2.y, cnoise(vec3(uv2 *rot(vTime * .5), 1.) * 10.));

  float starR = Star(uv2, sin(vTime) * 3.);

  // starR = fract(starR);

  float starG = Star(uv2 + .2, cos(vTime) * 3.);

  // starG = fract(starG);

  float starB = Star(uv2 -.2, .3);

  // starB = fract(starB);

  backgroundColor = vec3(starR * uValueA, starG * uValueB, starB * uValueD );


  float mStar = Star(uv, .8 * uValueC);
  mStar = fract(mStar * 4. * uValueC);
  backgroundColor = mix(backgroundColor, vec3(uValueA, uValueB, uValueC), mStar) ;


  // backgroundColor += star;

  // coswarp(backgroundColor, 3.);
  //
  //
  // backgroundColor.xyz += 8. * .1 * cos(3. * backgroundColor.yzx + vTime * .5) ;
  // backgroundColor = opCheapBend(backgroundColor);
  // backgroundColor = opTwist(backgroundColor);
  // backgroundColor.xyz += 8. * .05 * cos(11. * backgroundColor.yzx + vTime);


  // backgroundColor += displacement;
  vec3 lightPosition = vec3(2, 2, 7);
  vec3 col = vec3(0);
  vec3 ro = vec3(.4, .4, 18. ); // ray origin that represents camera position
  vec3 rd = normalize(vec3(uv, -1)); // ray direction

  vec4 co = rayMarch(ro, rd, MIN_DIST, MAX_DIST); // closest object

  if (co.x > MAX_DIST) {
    col = backgroundColor; // ray didn't hit anything
  } else {
    vec3 p = ro + rd * co.x; // point on sphere or floor we discovered from ray marching
    vec3 normal = calcNormal(p);

    vec3 lightDirection = normalize(lightPosition - p);

    // Calculate diffuse reflection by taking the dot product of
    // the normal and the light direction.
    float dif = clamp(dot(normal, lightDirection), 0.3, 1.);

    // Multiply the diffuse reflection value by an orange color and add a bit
    // of the background color to the sphere to blend it more with the background.
    // col = dif * co.yzw + backgroundColor * .1;
    col = dif * co.yzw ;
  }

  // Output to screen
  gl_FragColor = vec4(col, 1.0);
}
