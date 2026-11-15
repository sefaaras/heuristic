/* hold_epanet.c -- thin MEX bridge to the EPANET 2.2 toolkit for HOLD.
 *
 *   info     = hold_epanet('open', inpfile)
 *              opens the network (closing any other one) and returns
 *              info.length (pipe lengths, in link order of the pipes) and
 *              info.njunc (number of junctions)
 *   [p, err] = hold_epanet('solve', diameters)
 *              sets every pipe diameter (file units, mm for SI networks), runs
 *              ONE steady-state hydraulic solution and returns the pressure at
 *              every junction (m) and EPANET's return code (0 = clean,
 *              1..6 = warning, > 100 = error)
 *   hold_epanet('close')
 *
 * The DLL (epanet2.dll, OWA EPANET 2.2, MIT licence) is loaded by full path from
 * the folder this MEX file lives in, so nothing has to be on the system PATH.
 * The double-precision project API is used throughout; hydraulics are initialised
 * with flag 0 (results not saved), so a solve touches no file.
 *
 * Build: mex hold_epanet.c   (in this folder)
 */
#include "mex.h"
#include <windows.h>
#include <string.h>

typedef struct Project *EN_Project;
typedef int (__stdcall *t_create)(EN_Project *);
typedef int (__stdcall *t_delete)(EN_Project);
typedef int (__stdcall *t_open)(EN_Project, const char *, const char *, const char *);
typedef int (__stdcall *t_close)(EN_Project);
typedef int (__stdcall *t_openH)(EN_Project);
typedef int (__stdcall *t_initH)(EN_Project, int);
typedef int (__stdcall *t_runH)(EN_Project, long *);
typedef int (__stdcall *t_closeH)(EN_Project);
typedef int (__stdcall *t_getcount)(EN_Project, int, int *);
typedef int (__stdcall *t_getnodetype)(EN_Project, int, int *);
typedef int (__stdcall *t_getlinktype)(EN_Project, int, int *);
typedef int (__stdcall *t_getnodevalue)(EN_Project, int, int, double *);
typedef int (__stdcall *t_getlinkvalue)(EN_Project, int, int, double *);
typedef int (__stdcall *t_setlinkvalue)(EN_Project, int, int, double);
typedef int (__stdcall *t_setstatusreport)(EN_Project, int);

enum { EN_NODECOUNT = 0, EN_LINKCOUNT = 2, EN_JUNCTION = 0, EN_PIPE = 1,
       EN_DIAMETER = 0, EN_LENGTH = 1, EN_PRESSURE = 11 };

static HMODULE lib = NULL;
static t_create f_create; static t_delete f_delete; static t_open f_open;
static t_close f_close; static t_openH f_openH; static t_initH f_initH;
static t_runH f_runH; static t_closeH f_closeH; static t_getcount f_getcount;
static t_getnodetype f_getnodetype; static t_getlinktype f_getlinktype;
static t_getnodevalue f_getnodevalue; static t_getlinkvalue f_getlinkvalue;
static t_setlinkvalue f_setlinkvalue;
static t_setstatusreport f_setstatusreport;

static EN_Project ph = NULL;
static int *pipes = NULL, npipe = 0;      /* link indices of the pipes */
static int *juncs = NULL, njunc = 0;      /* node indices of the junctions */

static void close_network(void)
{
    if (ph) { f_closeH(ph); f_close(ph); f_delete(ph); ph = NULL; }
    if (pipes) { mxFree(pipes); pipes = NULL; }
    if (juncs) { mxFree(juncs); juncs = NULL; }
    npipe = njunc = 0;
}

static void at_exit(void)
{
    close_network();
    if (lib) { FreeLibrary(lib); lib = NULL; }
}

static FARPROC need(const char *name)
{
    FARPROC p = GetProcAddress(lib, name);
    if (!p) mexErrMsgIdAndTxt("hold_epanet:dll", "epanet2.dll has no %s.", name);
    return p;
}

static void load_library(void)
{
    char path[MAX_PATH];
    HMODULE self = NULL;
    char *slash;
    if (lib) return;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&load_library, &self);
    GetModuleFileNameA(self, path, MAX_PATH);
    slash = strrchr(path, '\\');
    if (slash) slash[1] = '\0';
    strncat(path, "epanet2.dll", MAX_PATH - strlen(path) - 1);
    lib = LoadLibraryA(path);
    if (!lib) mexErrMsgIdAndTxt("hold_epanet:dll", "Cannot load %s.", path);
    f_create = (t_create)need("EN_createproject");
    f_delete = (t_delete)need("EN_deleteproject");
    f_open = (t_open)need("EN_open");
    f_close = (t_close)need("EN_close");
    f_openH = (t_openH)need("EN_openH");
    f_initH = (t_initH)need("EN_initH");
    f_runH = (t_runH)need("EN_runH");
    f_closeH = (t_closeH)need("EN_closeH");
    f_getcount = (t_getcount)need("EN_getcount");
    f_getnodetype = (t_getnodetype)need("EN_getnodetype");
    f_getlinktype = (t_getlinktype)need("EN_getlinktype");
    f_getnodevalue = (t_getnodevalue)need("EN_getnodevalue");
    f_getlinkvalue = (t_getlinkvalue)need("EN_getlinkvalue");
    f_setlinkvalue = (t_setlinkvalue)need("EN_setlinkvalue");
    f_setstatusreport = (t_setstatusreport)need("EN_setstatusreport");
    mexAtExit(at_exit);
}

static void do_open(const char *inp, mxArray **out)
{
    int nn, nl, i, t, err;
    double *len;
    const char *fields[] = {"length", "njunc"};
    close_network();
    if ((err = f_create(&ph)) != 0) mexErrMsgIdAndTxt("hold_epanet:open", "EN_createproject: %d", err);
    /* report to the null device: warnings such as negative pressures would
       otherwise be written to stdout on every solve */
    if ((err = f_open(ph, inp, "NUL", "")) > 100) {
        f_delete(ph); ph = NULL;
        mexErrMsgIdAndTxt("hold_epanet:open", "EN_open(%s): error %d", inp, err);
    }
    f_setstatusreport(ph, 0);   /* EN_NO_REPORT */
    if ((err = f_openH(ph)) > 100) mexErrMsgIdAndTxt("hold_epanet:open", "EN_openH: error %d", err);
    f_getcount(ph, EN_NODECOUNT, &nn);
    f_getcount(ph, EN_LINKCOUNT, &nl);
    pipes = (int *)mxMalloc(nl * sizeof(int));
    juncs = (int *)mxMalloc(nn * sizeof(int));
    mexMakeMemoryPersistent(pipes);
    mexMakeMemoryPersistent(juncs);
    for (i = 1; i <= nl; i++) { f_getlinktype(ph, i, &t); if (t == EN_PIPE) pipes[npipe++] = i; }
    for (i = 1; i <= nn; i++) { f_getnodetype(ph, i, &t); if (t == EN_JUNCTION) juncs[njunc++] = i; }
    *out = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(*out, 0, "length", mxCreateDoubleMatrix(npipe, 1, mxREAL));
    len = mxGetPr(mxGetField(*out, 0, "length"));
    for (i = 0; i < npipe; i++) f_getlinkvalue(ph, pipes[i], EN_LENGTH, &len[i]);
    mxSetField(*out, 0, "njunc", mxCreateDoubleScalar(njunc));
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char cmd[16];
    if (nrhs < 1 || !mxIsChar(prhs[0])) mexErrMsgIdAndTxt("hold_epanet:usage", "First argument is a command.");
    mxGetString(prhs[0], cmd, sizeof(cmd));
    load_library();

    if (strcmp(cmd, "open") == 0) {
        char *inp;
        if (nrhs < 2 || !mxIsChar(prhs[1])) mexErrMsgIdAndTxt("hold_epanet:usage", "open needs an .inp file.");
        inp = mxArrayToString(prhs[1]);
        do_open(inp, &plhs[0]);
        mxFree(inp);
    } else if (strcmp(cmd, "solve") == 0) {
        const double *d;
        double *p;
        long t = 0;
        int i, err, worst = 0;
        if (!ph) mexErrMsgIdAndTxt("hold_epanet:state", "No network is open.");
        if (nrhs < 2 || mxGetNumberOfElements(prhs[1]) != (size_t)npipe)
            mexErrMsgIdAndTxt("hold_epanet:usage", "solve needs %d diameters.", npipe);
        d = mxGetPr(prhs[1]);
        for (i = 0; i < npipe; i++) {
            err = f_setlinkvalue(ph, pipes[i], EN_DIAMETER, d[i]);
            if (err > worst) worst = err;
        }
        if ((err = f_initH(ph, 0)) > worst) worst = err;
        if ((err = f_runH(ph, &t)) > worst) worst = err;
        plhs[0] = mxCreateDoubleMatrix(njunc, 1, mxREAL);
        p = mxGetPr(plhs[0]);
        for (i = 0; i < njunc; i++) f_getnodevalue(ph, juncs[i], EN_PRESSURE, &p[i]);
        if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(worst);
    } else if (strcmp(cmd, "close") == 0) {
        close_network();
    } else {
        mexErrMsgIdAndTxt("hold_epanet:usage", "Unknown command '%s'.", cmd);
    }
}
