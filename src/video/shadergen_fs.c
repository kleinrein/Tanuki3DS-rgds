#include "shadergen_fs.h"

#include <math.h>

#include "dynstring.h"

const char fs_header[] = R"(
#version 330 core

in vec4 color;
in vec2 texcoord0;
in vec2 texcoord1;
in vec2 texcoord2;
in float texcoordw;
in vec3 normal;
in vec3 tangent;
in vec3 view;

out vec4 fragclr;

uniform sampler2D tex0;
uniform sampler2DShadow tex0shadow;

uniform sampler2D tex1;
uniform sampler2D tex2;

uniform sampler1D tex3;

uniform sampler1DArray lightLuts;
uniform sampler1D fogLut;
uniform sampler1D proctexMapLut;
uniform sampler1D proctexNoiseLut;

struct Light {
    vec3 specular0;
    vec3 specular1;
    vec3 diffuse;
    vec3 ambient;
    vec3 vec;
    vec4 spotdir;
    float attn_bias;
    float attn_scale;
};

struct PTNoiseParam {
    float ampl;
    float phase;
    float freq;
    float pad;
};

layout (std140) uniform FragUniforms {
    vec4 tev_color[6];
    vec4 tev_buffer_color;

    PTNoiseParam noiseU;
    PTNoiseParam noiseV;

    Light light[8];
    vec3 ambient_color;
    vec4 fog_color;

    float shadowMax;
    float shadowRamp;
    float shadowBias;
    float alpharef;
};

#define LUT_D0 0
#define LUT_D1 1
#define LUT_FR 3
#define LUT_RB 4
#define LUT_RG 5
#define LUT_RR 6
#define LUT_SP_BASE 8
#define LUT_DA_BASE 16

)";

const u8 lightenvEnabledLuts[8] = {
    0xc5, 0xcc, 0xc3, 0x8b, 0xf7, 0xfd, 0xcf, 0xff,
};

const char* lutinput_str(int lutSel) {
    switch (lutSel) {
        case 0:
            return "dot(n, h)";
        case 1:
            return "dot(v, h)";
        case 2:
            return "dot(n, v)";
        case 3:
            return "dot(n, l)";
        case 4:
            return "dot(l, p)";
        case 5:
            return "dot(t, h_proj)";
        default:
            return "0";
    }
}

void write_lut_read(DynString* s, FragConfig* fcfg, int lutNum, char* name) {
    ds_printf(s, "texture(lightLuts, vec2(");
    if (fcfg->llutAbs & BIT(4 * lutNum + 1)) {
        // to ensure proper clamping we need to multiply by the size of the
        // texture
        ds_printf(s, "fract(clamp(%s*128,-127.5,127.5)/256)",
                  lutinput_str(fcfg->llutSel >> 4 * lutNum & 7));
    } else {
        ds_printf(s, "abs(%s)", lutinput_str(fcfg->llutSel >> 4 * lutNum & 7));
    }
    ds_printf(s, ", %s)).r", name);
    int scaleExp = (sbit(3))(fcfg->llutScale >> 4 * lutNum);
    if (scaleExp != 0) {
        float scale = powf(2, scaleExp);
        ds_printf(s, "*%.2f", scale);
    }
}

void write_lighting(DynString* s, FragConfig* fcfg) {
    ds_printf(s, "vec3 n = normalize(normal);\n");
    ds_printf(s, "vec3 t = normalize(tangent);\n");

    if (fcfg->lconfig0.bumpMode != 0) {
        ds_printf(s, "vec3 bumpVec = 2 * tex%dc.xyz - 1;\n",
                  fcfg->lconfig0.bumpTex);
        if (!fcfg->lconfig0.noRecalcBumpVec) {
            // recalcuate z value, if this is set then z is calculated to make a
            // unit vector
            ds_printf(
                s,
                "bumpVec.z = sqrt(max(1 - dot(bumpVec.xy, bumpVec.xy), 0));\n");
        }
        // reconstruct the tbn matrix from normal and tangent and
        // use it to rotate the bump map vector
        // pica uses quaternions which are translated into normal/tangent
        // vectors in the vertex shader
        ds_printf(s, "mat3 tbn = mat3(t, cross(n,t), n);\n");
        if (fcfg->lconfig0.bumpMode == 1) {
            ds_printf(s, "n = normalize(tbn * bumpVec);\n");
        }
        if (fcfg->lconfig0.bumpMode == 2) {
            ds_printf(s, "t = normalize(tbn * bumpVec);\n");
        }
    }

    ds_printf(s, "vec3 v = normalize(view);\n");

    if (fcfg->lconfig0.shadow) {
        if (fcfg->lconfig0.shadowInv) {
            ds_printf(s, "vec4 H = 1 - tex%dc;\n", fcfg->lconfig0.shadowTex);
        } else {
            ds_printf(s, "vec4 H = tex%dc;\n", fcfg->lconfig0.shadowTex);
        }
    }

    ds_printf(s, "vec3 l, h, h_proj, p;\n");
    ds_printf(s, "float dist;\n");

    int envLuts = lightenvEnabledLuts[fcfg->lconfig0.lightenv];
    int disabledLuts = fcfg->lconfig1.luts;
    int enabledLuts = envLuts & ~disabledLuts;

    int i; // need this after the loop for fresnel
    for (int _i = 0; _i < fcfg->numlights; _i++) {
        i = fcfg->lightPerm >> 4 * _i & 7;

        if (fcfg->light[i].config.directional) {
            ds_printf(s, "l = light[%d].vec;\n", i);
        } else {
            ds_printf(s, "l = view + light[%d].vec;\n", i);
        }
        ds_printf(s, "dist = length(l);\n");
        ds_printf(s, "l /= dist;\n");
        ds_printf(s, "h = normalize(l + v);\n");
        ds_printf(s, "h_proj = normalize(h - n * dot(n, h));\n");
        ds_printf(s, "p = normalize(light[%d].spotdir.xyz);\n", i);

        const char* ndotlstr = fcfg->light[i].config.twosided
                                   ? "abs(dot(n, l))"
                                   : "max(dot(n, l), 0)";

        ds_printf(s, "vec3 cp_%d = %s * light[%d].diffuse;\n", i, ndotlstr, i,
                  i);

        if (fcfg->light[i].config.use_g0 || fcfg->light[i].config.use_g1) {
            ds_printf(s, "float G_%d = %s/dot(l+v,l+v);\n", i, ndotlstr);
        }

        bool use_R = false;
        if (envLuts & BIT(LLUT_RR) &&
            (enabledLuts & (BIT(LLUT_RR) | BIT(LLUT_RG) | BIT(LLUT_RB)))) {
            use_R = true;
            ds_printf(s, "vec3 R_%d = vec3(", i);
            if (envLuts & BIT(LLUT_RG)) {
                if (enabledLuts & BIT(LLUT_RR)) {
                    write_lut_read(s, fcfg, LLUT_RR, "LUT_RR");
                } else {
                    ds_printf(s, "1");
                }
                ds_printf(s, ", ");
                if (enabledLuts & BIT(LLUT_RG)) {
                    write_lut_read(s, fcfg, LLUT_RG, "LUT_RG");
                } else {
                    ds_printf(s, "1");
                }
                ds_printf(s, ", ");
                if (enabledLuts & BIT(LLUT_RB)) {
                    write_lut_read(s, fcfg, LLUT_RB, "LUT_RB");
                } else {
                    ds_printf(s, "1");
                }
            } else {
                if (enabledLuts & BIT(LLUT_RR)) {
                    write_lut_read(s, fcfg, LLUT_RR, "LUT_RR");
                } else {
                    ds_printf(s, "1");
                }
            }
            ds_printf(s, ");\n");
        }

        ds_printf(s, "vec3 cs_%d = ", i);
        if (enabledLuts & BIT(LLUT_D0)) {
            write_lut_read(s, fcfg, LLUT_D0, "LUT_D0");
            ds_printf(s, " * ");
        }
        if (fcfg->light[i].config.use_g0) {
            ds_printf(s, "G_%d * ", i);
        }
        ds_printf(s, "light[%d].specular0 +\n", i);
        if (enabledLuts & BIT(LLUT_D1)) {
            write_lut_read(s, fcfg, LLUT_D1, "LUT_D1");
            ds_printf(s, " * ");
        }
        if (use_R) {
            ds_printf(s, "R_%d * ", i);
        }
        if (fcfg->light[i].config.use_g1) {
            ds_printf(s, "G_%d * ", i);
        }
        ds_printf(s, "light[%d].specular1;\n", i);

        if (!(fcfg->lconfig1.shadow & BIT(i))) {
            if (fcfg->lconfig0.shadowPrimary) {
                ds_printf(s, "cp_%d *= H.rgb;\n", i);
            }
            if (fcfg->lconfig0.shadowSecondary) {
                ds_printf(s, "cs_%d *= H.rgb;\n", i);
            }
        }

        ds_printf(s, "cp_%d += light[%d].ambient;\n", i, i);

        if (fcfg->lconfig0.clampHighlights) {
            ds_printf(s, "float f_%d = float(dot(n,l)>=0);\n", i);
            ds_printf(s, "cp_%d *= f_%d;\n", i, i);
            ds_printf(s, "cs_%d *= f_%d;\n", i, i);
        }

        if (envLuts & BIT(LLUT_SP) && !(fcfg->lconfig1.spotlight & BIT(i))) {
            ds_printf(s, "float S_%d = ", i);
            char buf[64];
            snprintf(buf, sizeof buf, "LUT_SP_BASE + %d", i);
            write_lut_read(s, fcfg, LLUT_SP, buf);
            ds_printf(s, ";\n");
            ds_printf(s, "cp_%d *= S_%d;\n", i, i);
            ds_printf(s, "cs_%d *= S_%d;\n", i, i);
        }
        if (envLuts & BIT(LLUT_DA) && !(fcfg->lconfig1.distattn & BIT(i))) {
            ds_printf(s,
                      "float A_%d = texture(lightLuts, vec2(\n"
                      "light[%d].attn_bias + light[%d].attn_scale * dist,\n"
                      "LUT_DA_BASE + %d)).r;\n",
                      i, i, i, i);
            ds_printf(s, "cp_%d *= A_%d;\n", i, i);
            ds_printf(s, "cs_%d *= A_%d;\n", i, i);
        }

        ds_printf(s, "lprimary.rgb += cp_%d;\n", i);
        ds_printf(s, "lsecondary.rgb += cs_%d;\n", i);
    }

    if (enabledLuts & BIT(LLUT_FR) &&
        (fcfg->lconfig0.frPrimary || fcfg->lconfig0.frSecondary)) {
        ds_printf(s, "float fr = ");
        write_lut_read(s, fcfg, LLUT_FR, "LUT_FR");
        ds_printf(s, ";\n");
        if (fcfg->lconfig0.frPrimary) {
            ds_printf(s, "lprimary.a = fr;\n");
        } else {
            ds_printf(s, "lprimary.a = 1;\n");
        }
        if (fcfg->lconfig0.frSecondary) {
            ds_printf(s, "lsecondary.a = fr;\n");
        } else {
            ds_printf(s, "lsecondary.a = 1;\n");
        }
    } else {
        ds_printf(s, "lprimary.a = 1;\n");
        ds_printf(s, "lsecondary.a = 1;\n");
    }

    if (fcfg->lconfig0.shadowAlpha) {
        ds_printf(s, "lprimary.a *= H.a;\n");
        ds_printf(s, "lsecondary.a *= H.a;\n");
    }

    ds_printf(s, "lprimary.rgb += ambient_color;\n");

    ds_printf(s, "lprimary = min(lprimary, 1);\n");
    ds_printf(s, "lsecondary = min(lsecondary, 1);\n");
}

void write_proctex_clamp(DynString* s, const char* v, u32 mode) {
    switch (mode) {
        case 0:
            ds_printf(s, "%s = mix(max(%s,0), 0, %s>=1);\n", v, v, v);
            break;
        case 1:
            ds_printf(s, "%s = clamp(%s,0,1);\n", v, v);
            break;
        case 2:
            ds_printf(s, "%s = fract(%s);\n", v, v);
            break;
        case 3:
            ds_printf(s, "%s = mix(fract(%s),fract(-%s),(int(%s) & 1) == 1);\n",
                      v, v, v, v);
            break;
        case 4:
            ds_printf(s, "%s = float(%s > 0.5)", v, v);
            break;
    }
}

const char* proctex_combiner_str(u32 combiner) {
    switch (combiner) {
        case 0:
            return "texcoord3.x";
        case 1:
            return "texcoord3.x * texcoord3.x";
        case 2:
            return "texcoord3.y";
        case 3:
            return "texcoord3.y * texcoord3.y";
        case 4:
            return "0.5 * (texcoord3.x + texcoord3.y)";
        case 5:
            return "0.5 * dot(texcoord3, texcoord3)";
        case 6:
            return "min(length(texcoord3), 1)";
        case 7:
            return "min(texcoord3.x, texcoord3.y)";
        case 8:
            return "max(texcoord3.x, texcoord3.y)";
        case 9:
            return "0.5 * (0.5 * (texcoord3.x + texcoord3.y) + "
                   "min(length(texcoord3), 1))";
        default:
            return "0";
    }
}

void write_proctex(DynString* s, FragConfig* fcfg) {
    ds_printf(s, "vec2 texcoord3 = texcoord%d;\n", fcfg->texconfig.tex3coord);

    if (fcfg->proctex.noise) {
        ds_printf(s, "texcoord3.x += noiseU.ampl * "
                     "texture(proctexNoiseLut, (texcoord3.x + noiseU.phase) / "
                     "noiseU.freq).r;\n");
        ds_printf(s, "texcoord3.y += noiseV.ampl * "
                     "texture(proctexNoiseLut, (texcoord3.y + noiseV.phase) / "
                     "noiseV.freq).r;\n");
    }

    write_proctex_clamp(s, "texcoord3.x", fcfg->proctex.clampU);
    write_proctex_clamp(s, "texcoord3.y", fcfg->proctex.clampV);

    ds_printf(s, "vec4 tex3c = texture(tex3, texture(proctexMapLut, %s).r);\n",
              proctex_combiner_str(fcfg->proctex.rgbCombiner));
    if (fcfg->proctex.separateAlpha) {
        ds_printf(s,
                  "tex3c.a = texture(tex3, texture(proctexMapLut, %s).g).a;\n",
                  proctex_combiner_str(fcfg->proctex.alphaCombiner));
    }
}

const char* tevsrc_str(int i, u32 tevsrc) {
    switch (tevsrc) {
        case TEVSRC_COLOR:
            return "color";
        case TEVSRC_LIGHT_PRIMARY:
            return "lprimary";
        case TEVSRC_LIGHT_SECONDARY:
            return "lsecondary";
        case TEVSRC_TEX0:
            return "tex0c";
        case TEVSRC_TEX1:
            return "tex1c";
        case TEVSRC_TEX2:
            return "tex2c";
        case TEVSRC_TEX3:
            return "tex3c";
        case TEVSRC_BUFFER:
            return "buf";
        case TEVSRC_CONSTANT: {
            // this is cursed
            switch (i) {
                case 0:
                    return "tev_color[0]";
                case 1:
                    return "tev_color[1]";
                case 2:
                    return "tev_color[2]";
                case 3:
                    return "tev_color[3]";
                case 4:
                    return "tev_color[4]";
                case 5:
                    return "tev_color[5]";
                default:
                    unreachable();
            }
        }
        case TEVSRC_PREVIOUS:
            return "cur";
        default:
            return "vec4(0)";
    }
}

void write_operand_rgb(DynString* s, const char* srcstr, u32 op) {
    switch (op) {
        case 0:
            ds_printf(s, "%s.rgb", srcstr);
            break;
        case 1:
            ds_printf(s, "(1 - %s.rgb)", srcstr);
            break;
        case 2:
            ds_printf(s, "%s.aaa", srcstr);
            break;
        case 3:
            ds_printf(s, "(1 - %s.aaa)", srcstr);
            break;
        case 4:
            ds_printf(s, "%s.rrr", srcstr);
            break;
        case 5:
            ds_printf(s, "(1 - %s.rrr)", srcstr);
            break;
        case 8:
            ds_printf(s, "%s.ggg", srcstr);
            break;
        case 9:
            ds_printf(s, "(1 - %s.ggg)", srcstr);
            break;
        case 12:
            ds_printf(s, "%s.bbb", srcstr);
            break;
        case 13:
            ds_printf(s, "(1 - %s.bbb)", srcstr);
            break;
        default:
            ds_printf(s, "%s.rgb", srcstr);
    }
}

void write_operand_a(DynString* s, const char* srcstr, u32 op) {
    switch (op) {
        case 0:
            ds_printf(s, "%s.a", srcstr);
            break;
        case 1:
            ds_printf(s, "(1 - %s.a)", srcstr);
            break;
        case 2:
            ds_printf(s, "%s.r", srcstr);
            break;
        case 3:
            ds_printf(s, "(1 - %s.r)", srcstr);
            break;
        case 4:
            ds_printf(s, "%s.g", srcstr);
            break;
        case 5:
            ds_printf(s, "(1 - %s.g)", srcstr);
            break;
        case 6:
            ds_printf(s, "%s.b", srcstr);
            break;
        case 7:
            ds_printf(s, "(1 - %s.b)", srcstr);
            break;
        default:
            ds_printf(s, "%s.a", srcstr);
    }
}

void write_combiner_rgb(DynString* s, FragConfig* fcfg, int i) {
#define SRC(n)                                                                 \
    write_operand_rgb(s, tevsrc_str(i, fcfg->tev[i].rgb.src##n),               \
                      fcfg->tev[i].rgb.op##n)
    switch (fcfg->tev[i].rgb.combiner) {
        case 0:
            SRC(0);
            break;
        case 1:
            SRC(0);
            ds_printf(s, " * ");
            SRC(1);
            break;
        case 2:
            ds_printf(s, "min(");
            SRC(0);
            ds_printf(s, " + ");
            SRC(1);
            ds_printf(s, ", 1)");
            break;
        case 3:
            ds_printf(s, "clamp(");
            SRC(0);
            ds_printf(s, " + ");
            SRC(1);
            ds_printf(s, " - 0.5, 0, 1)");
            break;
        case 4:
            ds_printf(s, "mix(");
            SRC(1);
            ds_printf(s, ", ");
            SRC(0);
            ds_printf(s, ", ");
            SRC(2);
            ds_printf(s, ")");
            break;
        case 5:
            ds_printf(s, "max(");
            SRC(0);
            ds_printf(s, " - ");
            SRC(1);
            ds_printf(s, ", 0)");
            break;
        case 6:
        case 7:
            ds_printf(s, "max(vec3(4 * dot(");
            SRC(0);
            ds_printf(s, " - 0.5, ");
            SRC(1);
            ds_printf(s, " - 0.5)), 0)");
            break;
        case 8:
            ds_printf(s, "min(");
            SRC(0);
            ds_printf(s, " * ");
            SRC(1);
            ds_printf(s, " + ");
            SRC(2);
            ds_printf(s, ", 1)");
            break;
        case 9:
            ds_printf(s, "min(");
            SRC(0);
            ds_printf(s, " + ");
            SRC(1);
            ds_printf(s, ", 1) * ");
            SRC(2);
            break;
        default:
            SRC(0);
            break;
    }
#undef SRC
}

void write_combiner_a(DynString* s, FragConfig* fcfg, int i) {
#define SRC(n)                                                                 \
    write_operand_a(s, tevsrc_str(i, fcfg->tev[i].a.src##n),                   \
                    fcfg->tev[i].a.op##n)
    switch (fcfg->tev[i].a.combiner) {
        case 0:
            SRC(0);
            break;
        case 1:
            SRC(0);
            ds_printf(s, " * ");
            SRC(1);
            break;
        case 2:
            ds_printf(s, "min(");
            SRC(0);
            ds_printf(s, " + ");
            SRC(1);
            ds_printf(s, ", 1)");
            break;
        case 3:
            ds_printf(s, "clamp(");
            SRC(0);
            ds_printf(s, " + ");
            SRC(1);
            ds_printf(s, " - 0.5, 0, 1)");
            break;
        case 4:
            ds_printf(s, "mix(");
            SRC(1);
            ds_printf(s, ", ");
            SRC(0);
            ds_printf(s, ", ");
            SRC(2);
            ds_printf(s, ")");
            break;
        case 5:
            ds_printf(s, "max(");
            SRC(0);
            ds_printf(s, " - ");
            SRC(1);
            ds_printf(s, ", 0)");
            break;
        case 6:
        case 7:
            ds_printf(s, "max(4 * (");
            SRC(0);
            ds_printf(s, " - 0.5) * (");
            SRC(1);
            ds_printf(s, " - 0.5), 0)");
            break;
        case 8:
            ds_printf(s, "min(");
            SRC(0);
            ds_printf(s, " * ");
            SRC(1);
            ds_printf(s, " + ");
            SRC(2);
            ds_printf(s, ", 1)");
            break;
        case 9:
            ds_printf(s, "min(");
            SRC(0);
            ds_printf(s, " + ");
            SRC(1);
            ds_printf(s, ", 1) * ");
            SRC(2);
            break;
        default:
            SRC(0);
            break;
    }
#undef SRC
}

const char* alphatest_str(int alphafunc) {
    switch (alphafunc) {
        case 0:
            return "false";
        case 1:
            return "true";
        case 2:
            return "(fragclr.a == alpharef)";
        case 3:
            return "(fragclr.a != alpharef)";
        case 4:
            return "(fragclr.a < alpharef)";
        case 5:
            return "(fragclr.a <= alpharef)";
        case 6:
            return "(fragclr.a > alpharef)";
        case 7:
            return "(fragclr.a >= alpharef)";
        default:
            return "true";
    }
}

// ---- uber shader: the configuration the generated shaders bake in, read from
// a uniform block instead. Keep it in step with shader_gen_fs() below.
const char fs_uber_body[] = R"(
struct TevSide {
    int src0; int op0; int src1; int op1; int src2; int op2;
    int combiner; float scale;
};
struct Tev { TevSide rgb; TevSide a; };

layout (std140) uniform FragConfig {
    Tev tev[6];
    uint c_tevbuf;
    uint c_texconfig;
    int c_tex0type;
    int c_shadowPerspective;
    uint c_proctex;
    int c_pad0; int c_pad1; int c_pad2;
    uvec4 c_light[8];
    int c_numlights;
    uint c_lconfig0;
    uint c_lconfig1;
    uint c_llutAbs;
    uint c_llutSel;
    uint c_llutScale;
    uint c_lightPerm;
    int c_lightDisable;
    int c_fragOp;
    int c_alphatest;
    int c_alphafunc;
};

#define BITS(v, s, n) (((v) >> uint(s)) & ((1u << uint(n)) - 1u))
#define BITSET(v, s) ((((v) >> uint(s)) & 1u) != 0u)

const uint envLutsTab[8] = uint[](0xc5u, 0xccu, 0xc3u, 0x8bu, 0xf7u, 0xfdu, 0xcfu, 0xffu);

vec4 tex0c, tex1c, tex2c, tex3c, lprimary, lsecondary, cur, buf;
vec3 n, t, v, l, h, h_proj, p;

vec4 texsel(uint i) {
    if (i == 0u) return tex0c;
    if (i == 1u) return tex1c;
    if (i == 2u) return tex2c;
    return tex3c;
}

float lutinput(uint sel) {
    if (sel == 0u) return dot(n, h);
    if (sel == 1u) return dot(v, h);
    if (sel == 2u) return dot(n, v);
    if (sel == 3u) return dot(n, l);
    if (sel == 4u) return dot(l, p);
    if (sel == 5u) return dot(t, h_proj);
    return 0.0;
}

float lutread(int lutNum, float layer) {
    float x = lutinput(BITS(c_llutSel, 4 * lutNum, 3));
    float c = BITSET(c_llutAbs, 4 * lutNum + 1)
        ? fract(clamp(x * 128.0, -127.5, 127.5) / 256.0) : abs(x);
    float r = texture(lightLuts, vec2(c, layer)).r;
    int e = int(BITS(c_llutScale, 4 * lutNum, 3));
    if (e >= 4) e -= 8;
    if (e != 0) r *= exp2(float(e));
    return r;
}

void lighting() {
    n = normalize(normal);
    t = normalize(tangent);
    uint bumpMode = BITS(c_lconfig0, 28, 2);
    if (bumpMode != 0u) {
        vec3 bumpVec = 2.0 * texsel(BITS(c_lconfig0, 22, 2)).xyz - 1.0;
        if (!BITSET(c_lconfig0, 30)) {
            bumpVec.z = sqrt(max(1.0 - dot(bumpVec.xy, bumpVec.xy), 0.0));
        }
        mat3 tbn = mat3(t, cross(n, t), n);
        if (bumpMode == 1u) n = normalize(tbn * bumpVec);
        if (bumpMode == 2u) t = normalize(tbn * bumpVec);
    }
    v = normalize(view);

    vec4 H = vec4(1.0);
    if (BITSET(c_lconfig0, 0)) {
        H = texsel(BITS(c_lconfig0, 24, 2));
        if (BITSET(c_lconfig0, 18)) H = 1.0 - H;
    }

    l = vec3(0.0); h = vec3(0.0); h_proj = vec3(0.0); p = vec3(0.0);
    float dist = 0.0;

    uint envLuts = envLutsTab[BITS(c_lconfig0, 4, 3)];
    uint enabledLuts = envLuts & ~BITS(c_lconfig1, 16, 8);
    bool use_R = BITSET(envLuts, 6) && (enabledLuts & 0x70u) != 0u;

    for (int k = 0; k < c_numlights; k++) {
        int i = int(BITS(c_lightPerm, 4 * k, 3));
        uint lc = c_light[i].x;

        if (BITSET(lc, 0)) l = light[i].vec;
        else l = view + light[i].vec;
        dist = length(l);
        l /= dist;
        h = normalize(l + v);
        h_proj = normalize(h - n * dot(n, h));
        p = normalize(light[i].spotdir.xyz);

        float ndotl = BITSET(lc, 1) ? abs(dot(n, l)) : max(dot(n, l), 0.0);
        vec3 cp = ndotl * light[i].diffuse;
        float G = ndotl / dot(l + v, l + v);

        vec3 R = vec3(1.0);
        if (use_R) {
            if (BITSET(envLuts, 5)) {
                if (BITSET(enabledLuts, 6)) R.r = lutread(6, 6.0);
                if (BITSET(enabledLuts, 5)) R.g = lutread(5, 5.0);
                if (BITSET(enabledLuts, 4)) R.b = lutread(4, 4.0);
            } else if (BITSET(enabledLuts, 6)) {
                R = vec3(lutread(6, 6.0));
            }
        }

        vec3 s0 = light[i].specular0;
        if (BITSET(enabledLuts, 0)) s0 *= lutread(0, 0.0);
        if (BITSET(lc, 2)) s0 *= G;
        vec3 s1 = light[i].specular1;
        if (BITSET(enabledLuts, 1)) s1 *= lutread(1, 1.0);
        if (use_R) s1 *= R;
        if (BITSET(lc, 3)) s1 *= G;
        vec3 cs = s0 + s1;

        if (!BITSET(BITS(c_lconfig1, 0, 8), i)) {
            if (BITSET(c_lconfig0, 16)) cp *= H.rgb;
            if (BITSET(c_lconfig0, 17)) cs *= H.rgb;
        }

        cp += light[i].ambient;

        if (BITSET(c_lconfig0, 27)) {
            float f = float(dot(n, l) >= 0.0);
            cp *= f;
            cs *= f;
        }
        if (BITSET(envLuts, 2) && !BITSET(BITS(c_lconfig1, 8, 8), i)) {
            float S = lutread(2, 8.0 + float(i));
            cp *= S;
            cs *= S;
        }
        if (BITSET(envLuts, 7) && !BITSET(BITS(c_lconfig1, 24, 8), i)) {
            float A = texture(lightLuts, vec2(
                light[i].attn_bias + light[i].attn_scale * dist,
                16.0 + float(i))).r;
            cp *= A;
            cs *= A;
        }
        lprimary.rgb += cp;
        lsecondary.rgb += cs;
    }

    lprimary.a = 1.0;
    lsecondary.a = 1.0;
    if (BITSET(enabledLuts, 3) && (BITSET(c_lconfig0, 2) || BITSET(c_lconfig0, 3))) {
        float fr = lutread(3, 3.0);
        if (BITSET(c_lconfig0, 2)) lprimary.a = fr;
        if (BITSET(c_lconfig0, 3)) lsecondary.a = fr;
    }
    if (BITSET(c_lconfig0, 19)) {
        lprimary.a *= H.a;
        lsecondary.a *= H.a;
    }
    lprimary.rgb += ambient_color;
    lprimary = min(lprimary, 1.0);
    lsecondary = min(lsecondary, 1.0);
}

float ptclamp(float x, uint mode) {
    if (mode == 0u) return x >= 1.0 ? 0.0 : max(x, 0.0);
    if (mode == 1u) return clamp(x, 0.0, 1.0);
    if (mode == 2u) return fract(x);
    if (mode == 3u) return (int(x) & 1) == 1 ? fract(-x) : fract(x);
    if (mode == 4u) return float(x > 0.5);
    return x;
}

float ptcombine(vec2 c, uint comb) {
    if (comb == 0u) return c.x;
    if (comb == 1u) return c.x * c.x;
    if (comb == 2u) return c.y;
    if (comb == 3u) return c.y * c.y;
    if (comb == 4u) return 0.5 * (c.x + c.y);
    if (comb == 5u) return 0.5 * dot(c, c);
    if (comb == 6u) return min(length(c), 1.0);
    if (comb == 7u) return min(c.x, c.y);
    if (comb == 8u) return max(c.x, c.y);
    if (comb == 9u) return 0.5 * (0.5 * (c.x + c.y) + min(length(c), 1.0));
    return 0.0;
}

vec4 proctex() {
    uint tc = BITS(c_texconfig, 8, 2);
    vec2 c = tc == 0u ? texcoord0 : tc == 1u ? texcoord1 : texcoord2;
    if (BITSET(c_proctex, 15)) {
        c.x += noiseU.ampl * texture(proctexNoiseLut, (c.x + noiseU.phase) / noiseU.freq).r;
        c.y += noiseV.ampl * texture(proctexNoiseLut, (c.y + noiseV.phase) / noiseV.freq).r;
    }
    c.x = ptclamp(c.x, BITS(c_proctex, 0, 3));
    c.y = ptclamp(c.y, BITS(c_proctex, 3, 3));
    vec4 r = texture(tex3, texture(proctexMapLut, ptcombine(c, BITS(c_proctex, 6, 4))).r);
    if (BITSET(c_proctex, 14)) {
        r.a = texture(tex3, texture(proctexMapLut, ptcombine(c, BITS(c_proctex, 10, 4))).g).a;
    }
    return r;
}

vec4 tevsrc(int i, int src) {
    if (src == 0) return color;
    if (src == 1) return lprimary;
    if (src == 2) return lsecondary;
    if (src == 3) return tex0c;
    if (src == 4) return tex1c;
    if (src == 5) return tex2c;
    if (src == 6) return tex3c;
    if (src == 13) return buf;
    if (src == 14) return tev_color[i];
    if (src == 15) return cur;
    return vec4(0.0);
}

vec3 oprgb(vec4 s, int op) {
    if (op == 1) return 1.0 - s.rgb;
    if (op == 2) return s.aaa;
    if (op == 3) return 1.0 - s.aaa;
    if (op == 4) return s.rrr;
    if (op == 5) return 1.0 - s.rrr;
    if (op == 8) return s.ggg;
    if (op == 9) return 1.0 - s.ggg;
    if (op == 12) return s.bbb;
    if (op == 13) return 1.0 - s.bbb;
    return s.rgb;
}

float opa(vec4 s, int op) {
    if (op == 1) return 1.0 - s.a;
    if (op == 2) return s.r;
    if (op == 3) return 1.0 - s.r;
    if (op == 4) return s.g;
    if (op == 5) return 1.0 - s.g;
    if (op == 6) return s.b;
    if (op == 7) return 1.0 - s.b;
    return s.a;
}

vec3 combrgb(int i) {
    int c = tev[i].rgb.combiner;
    vec3 s0 = oprgb(tevsrc(i, tev[i].rgb.src0), tev[i].rgb.op0);
    if (c == 0 || c > 9) return s0;
    vec3 s1 = oprgb(tevsrc(i, tev[i].rgb.src1), tev[i].rgb.op1);
    if (c == 1) return s0 * s1;
    if (c == 2) return min(s0 + s1, 1.0);
    if (c == 3) return clamp(s0 + s1 - 0.5, 0.0, 1.0);
    if (c == 5) return max(s0 - s1, 0.0);
    if (c == 6 || c == 7) return max(vec3(4.0 * dot(s0 - 0.5, s1 - 0.5)), 0.0);
    vec3 s2 = oprgb(tevsrc(i, tev[i].rgb.src2), tev[i].rgb.op2);
    if (c == 4) return mix(s1, s0, s2);
    if (c == 8) return min(s0 * s1 + s2, 1.0);
    return min(s0 + s1, 1.0) * s2;
}

float comba(int i) {
    int c = tev[i].a.combiner;
    float s0 = opa(tevsrc(i, tev[i].a.src0), tev[i].a.op0);
    if (c == 0 || c > 9) return s0;
    float s1 = opa(tevsrc(i, tev[i].a.src1), tev[i].a.op1);
    if (c == 1) return s0 * s1;
    if (c == 2) return min(s0 + s1, 1.0);
    if (c == 3) return clamp(s0 + s1 - 0.5, 0.0, 1.0);
    if (c == 5) return max(s0 - s1, 0.0);
    if (c == 6 || c == 7) return max(4.0 * (s0 - 0.5) * (s1 - 0.5), 0.0);
    float s2 = opa(tevsrc(i, tev[i].a.src2), tev[i].a.op2);
    if (c == 4) return mix(s1, s0, s2);
    if (c == 8) return min(s0 * s1 + s2, 1.0);
    return min(s0 + s1, 1.0) * s2;
}

bool alphatest_pass(float a) {
    if (c_alphafunc == 0) return false;
    if (c_alphafunc == 2) return a == alpharef;
    if (c_alphafunc == 3) return a != alpharef;
    if (c_alphafunc == 4) return a < alpharef;
    if (c_alphafunc == 5) return a <= alpharef;
    if (c_alphafunc == 6) return a > alpharef;
    if (c_alphafunc == 7) return a >= alpharef;
    return true;
}

void main() {
    tex0c = vec4(0.0);
    if (BITSET(c_texconfig, 0)) {
        // a shadow map (type 2) needs the other sampler on the same unit, which
        // one program cannot have: the renderer waits for the exact program
        if (c_tex0type == 0) tex0c = texture(tex0, texcoord0);
        else if (c_tex0type == 1 || c_tex0type == 4) tex0c = vec4(1.0);
        else if (c_tex0type == 3) tex0c = textureProj(tex0, vec3(texcoord0, texcoordw));
    }
    tex1c = BITSET(c_texconfig, 1) ? texture(tex1, texcoord1) : vec4(0.0);
    tex2c = BITSET(c_texconfig, 2)
        ? texture(tex2, BITSET(c_texconfig, 13) ? texcoord1 : texcoord2) : vec4(0.0);
    tex3c = BITSET(c_texconfig, 10) ? proctex() : vec4(0.0);

    lprimary = vec4(0.0);
    lsecondary = vec4(0.0);
    if (c_lightDisable == 0) lighting();

    cur = vec4(0.0);
    buf = tev_buffer_color;
    for (int i = 0; i < 6; i++) {
        vec4 tmp;
        tmp.rgb = combrgb(i);
        if (tev[i].rgb.combiner == 7) tmp.a = tmp.r;
        else tmp.a = comba(i);
        if (i > 0) {
            if (BITSET(BITS(c_tevbuf, 8, 4), i - 1)) buf.rgb = cur.rgb;
            if (BITSET(BITS(c_tevbuf, 12, 4), i - 1)) buf.a = cur.a;
        }
        cur = tmp;
        if (tev[i].rgb.scale != 1.0) cur.rgb = min(cur.rgb * tev[i].rgb.scale, 1.0);
        if (tev[i].a.scale != 1.0) cur.a = min(cur.a * tev[i].a.scale, 1.0);
    }

    fragclr = clamp(cur, 0.0, 1.0);

    if (c_alphatest != 0 && !alphatest_pass(fragclr.a)) discard;

    if (BITS(c_tevbuf, 0, 3) == 5u) {
        float fc = gl_FragCoord.z * 128.0 / 129.0 + 0.5 / 129.0;
        if (BITSET(c_tevbuf, 16)) fc = 1.0 - fc;
        float fog = texture(fogLut, fc).r;
        fragclr.rgb = mix(fog_color.rgb, fragclr.rgb, fog);
    }
}
)";

char* shader_gen_uber_fs(void) {
    DynString s;
    ds_init(&s, 32768);
    ds_printf(&s, "%s", fs_header);
    ds_printf(&s, "%s", fs_uber_body);
    return s.str;
}

char* shader_gen_fs(FragConfig* fcfg) {
    DynString s;
    ds_init(&s, 8192);

    ds_printf(&s, fs_header);

    ds_printf(&s, "void main() {\n");

    if (fcfg->texconfig.tex0enable) {
        switch (fcfg->tex0type) {
            case 0: // normal
                ds_printf(&s, "vec4 tex0c = texture(tex0, texcoord0);\n");
                break;
            case 2: // shadow map
                // shadow map sampling
                // texcoord0.uvw is the fragment position in light space
                // we read out the depth from the shadow map and compare
                // with texcoord0.w which is fragment depth in light space
                // to determine if it is in the shadow
                // previously we did this by hand but now are using
                // sampler2DShadow which does all this for us
                ds_printf(&s,
                          "vec4 tex0c = vec4("
                          "1 - texture(tex0shadow, vec3(texcoord0%s, "
                          "texcoordw-shadowBias)));\n",
                          fcfg->shadowPerspective ? "/texcoordw" : "");
                break;
            case 1: // cube
            case 4: // shadow cube
                lwarnonce("cube map texture");
                ds_printf(&s, "vec4 tex0c = vec4(1);\n");
                break;
            case 3: // projection
                ds_printf(&s, "vec4 tex0c = textureProj(tex0, vec3(texcoord0, "
                              "texcoordw));\n");
                break;
            default:
                lwarnonce("unknown texture type %d", fcfg->tex0type);
                ds_printf(&s, "vec4 tex0c = vec4(0);\n");
        }
    } else ds_printf(&s, "vec4 tex0c = vec4(0);\n");
    if (fcfg->texconfig.tex1enable) {
        ds_printf(&s, "vec4 tex1c = texture(tex1, texcoord1);\n");
    } else ds_printf(&s, "vec4 tex1c = vec4(0);\n");
    if (fcfg->texconfig.tex2enable) {
        ds_printf(&s, "vec4 tex2c = texture(tex2, texcoord%d);\n",
                  fcfg->texconfig.tex2coord ? 1 : 2);
    } else ds_printf(&s, "vec4 tex2c = vec4(0);\n");
    if (fcfg->texconfig.tex3enable) {
        write_proctex(&s, fcfg);
    } else ds_printf(&s, "vec4 tex3c = vec4(0);\n");

    ds_printf(&s, "vec4 lprimary = vec4(0);\n");
    ds_printf(&s, "vec4 lsecondary = vec4(0);\n");
    if (!fcfg->lightDisable) {
        write_lighting(&s, fcfg);
    }

    ds_printf(&s, R"(
vec4 cur = vec4(0);
vec4 buf = tev_buffer_color;
vec4 tmp;
)");

    for (int i = 0; i < 6; i++) {
        // check for do nothing stage
        bool skiprgb = fcfg->tev[i].rgb.combiner == 0 &&
                       fcfg->tev[i].rgb.op0 == 0 &&
                       fcfg->tev[i].rgb.src0 == TEVSRC_PREVIOUS;
        bool skipa = fcfg->tev[i].a.combiner == 0 && fcfg->tev[i].a.op0 == 0 &&
                     fcfg->tev[i].a.src0 == TEVSRC_PREVIOUS;
        if (!(skiprgb && skipa)) {
            ds_printf(&s, "tmp.rgb = ");
            write_combiner_rgb(&s, fcfg, i);
            ds_printf(&s, ";\n");
            if (fcfg->tev[i].rgb.combiner == 7) {
                ds_printf(&s, "tmp.a = tmp.r;\n");
            } else {
                ds_printf(&s, "tmp.a = ");
                write_combiner_a(&s, fcfg, i);
                ds_printf(&s, ";\n");
            }
        }

        if (i > 0) {
            // buffer update should happen in parallel
            // with the tev combiner
            // input from previous stage and not visible until next stage
            if (fcfg->tev_buffer.update_rgb & BIT(i - 1)) {
                ds_printf(&s, "buf.rgb = cur.rgb;\n");
            }
            if (fcfg->tev_buffer.update_alpha & BIT(i - 1)) {
                ds_printf(&s, "buf.a = cur.a;\n");
            }
        }

        if (!(skiprgb && skipa)) {
            ds_printf(&s, "cur = tmp;\n");
        }

        if (fcfg->tev[i].rgb.scale != 1.0f) {
            ds_printf(&s, "cur.rgb = min(cur.rgb * %.0f, 1);\n",
                      fcfg->tev[i].rgb.scale);
        }
        if (fcfg->tev[i].a.scale != 1.0f) {
            ds_printf(&s, "cur.a = min(cur.a * %.0f, 1);\n",
                      fcfg->tev[i].a.scale);
        }
    }

    ds_printf(&s, "fragclr = clamp(cur, 0, 1);\n");

    if (fcfg->alphatest) {
        ds_printf(&s, "if (!%s) discard;\n", alphatest_str(fcfg->alphafunc));
    }

    if (fcfg->tev_buffer.fogmode == 5) {
        ds_printf(&s, "float fog = texture(fogLut, ");
        if (fcfg->tev_buffer.zflip) {
            ds_printf(&s, "1 - ");
        }
        // gl will take coordinate n.5 as the value at n in the texture
        // and coordinate n is actually interpolated between n-1,n
        // but we want to shift over by 0.5
        // coordinate 0/128 -> 0.5, ... 127/128 -> 127.5, 128/128 -> 128.5
        // for proper fog we recorrect the coordinate (it is quite sensitive to
        // this)
        ds_printf(&s, "(gl_FragCoord.z*128.f/129+0.5f/129)).r;\n");
        ds_printf(&s, "fragclr.rgb = mix(fog_color.rgb, fragclr.rgb, fog);\n");
    }

    ds_printf(&s, "}\n");

    return s.str;
}
