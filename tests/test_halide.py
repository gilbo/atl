
import unittest
from ATL.halide import *
import time

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestHalideWrapper(unittest.TestCase):

    def new_buffer(self, x,y,w,h):
        arr   = ((ctypes.c_int * w) * h)()
        p_arr = ctypes.cast( arr, ctypes.POINTER(ctypes.c_ubyte) )
        
        out_buf  = halide_buffer_t()
        out_buf.device              = 0
        out_buf.device_interface    = None
        out_buf.host                = p_arr
        out_buf.flags               = 0
        out_buf.type                = halide_type_t(C.type_int,32,1)
        out_buf.dimensions          = 2
        out_buf.dim                 = (halide_dimension_t * 2)()
        out_buf.dim[0] = halide_dimension_t(x,w,1,0)
        out_buf.dim[1] = halide_dimension_t(y,h,w,0)
        out_buf.padding             = None
        
        return out_buf, arr

    def test_run_tut1(self):
        print()
        gradient = C.hwrap_new_func(b"gradient")
        x        = C.hwrap_new_var(b"x")
        y        = C.hwrap_new_var(b"y")
        c        = C.hwrap_new_param(b"c",halide_type_t(C.type_int,32,1))
        C.hwrap_set_param(c,ctypes.byref(ctypes.c_int(2)))

        # e = c * (x + y)
        e_x      = C.hwrap_var_to_expr(x)
        e_y      = C.hwrap_var_to_expr(y)
        e        = C.hwrap_add(e_x,e_y)
        e_c      = C.hwrap_param_to_expr(c)
        e        = C.hwrap_mul(e_c,e)
        
        # gradient(x,y) = e
        idx      = (hw_var_t * 2)(x,y)
        C.hwrap_pure_def(gradient,2,idx,e)
        
        # set up the buffer
        W,H      = 1000,1000
        i32E     = C.hwrap_i32_to_expr
        C.hwrap_set_func_bound_estimate(gradient,0,i32E(0),i32E(W))
        C.hwrap_set_func_bound_estimate(gradient,1,i32E(0),i32E(H))
        buf, arr = self.new_buffer(0,0,W,H)

        #C.hwrap_func_print_loop_nest(gradient)
        #C.hwrap_autoschedule_func(gradient)
            
        # run tests
        t0       = time.perf_counter()
        C.hwrap_realize_func(gradient,buf)
        t1       = time.perf_counter()
        C.hwrap_realize_func(gradient,buf)
        t2       = time.perf_counter()
        for k in range(0,10):
            C.hwrap_realize_func(gradient,buf)
        t3       = time.perf_counter()
        print('times', t1-t0,t2-t1,t3-t2)
        
        # test the result
        for j in range(0,600):
            for i in range(0,800):
                if arr[j][i] != 2*(i + j):
                    print(f"Something went wrong!\n"+
                          f"Pixel {i}, {j} was supposed to be {2*(i+j)},"
                          f"but instead it's {arr[j][i]}")
        
        print("Success!")

    def build_blur(self, orig,x,y):
        i32E     = C.hwrap_i32_to_expr
        varE     = C.hwrap_var_to_expr
        fEst     = C.hwrap_set_func_bound_estimate
        iEst     = C.hwrap_set_img_bound_estimate
        
        # new func defs
        blur_x   = C.hwrap_new_func(b"blur_x")
        blur_y   = C.hwrap_new_func(b"blur_y")
        
        e_x, e_y = varE(x), varE(y)
        
        # expressions and statements
        # blur_x(x,y) = (orig(x-1,y) + 2*orig(x,y) + orig(x+1,y))/4
        x_m1     = C.hwrap_sub(varE(x),i32E(1))
        x_p1     = C.hwrap_add(varE(x),i32E(1))
        o_mid    = C.hwrap_access_func(orig,2,(hw_expr_t*2)(e_x,e_y))
        o_left   = C.hwrap_access_func(orig,2,(hw_expr_t*2)(x_m1,e_y))
        o_right  = C.hwrap_access_func(orig,2,(hw_expr_t*2)(x_p1,e_y))
        bx_sum   = C.hwrap_add(C.hwrap_add( o_left,
                                            C.hwrap_mul(i32E(2),o_mid)),
                                o_right)
        bx_avg   = C.hwrap_div(bx_sum,i32E(4))
        C.hwrap_pure_def(blur_x,2,(hw_var_t*2)(x,y),bx_avg)
        
        # blur_y(x,y) = (blur_x(x,y-1) + 2*blur_x(x,y) + blur_x(x,y+1))/4
        y_m1     = C.hwrap_sub(e_y,i32E(1))
        y_p1     = C.hwrap_add(e_y,i32E(1))
        o_mid    = C.hwrap_access_func(blur_x,2,(hw_expr_t*2)(e_x,e_y))
        o_top    = C.hwrap_access_func(blur_x,2,(hw_expr_t*2)(e_x,y_m1))
        o_bot    = C.hwrap_access_func(blur_x,2,(hw_expr_t*2)(e_x,y_p1))
        by_sum   = C.hwrap_add(C.hwrap_add( o_top,
                                            C.hwrap_mul(i32E(2),o_mid)),
                                o_bot)
        by_avg   = C.hwrap_div(by_sum,i32E(4))
        C.hwrap_pure_def(blur_y,2,(hw_var_t*2)(x,y),by_avg)
        
        return blur_x, blur_y
    
    def blur_test_0(self, use_auto=False):
        i32E     = C.hwrap_i32_to_expr
        varE     = C.hwrap_var_to_expr
        fEst     = C.hwrap_set_func_bound_estimate
        iEst     = C.hwrap_set_img_bound_estimate
        
        # set up the buffers
        W,H          = 1000,1000
        bufI, arrI   = self.new_buffer(0,0,W,H)
        bufO, arrO   = self.new_buffer(1,1,W-2,H-2)
        
        # set up the input image parameter
        inImg    = C.hwrap_new_img(b"inImg",2,halide_type_t(C.type_int,32,1))
        C.hwrap_set_img(inImg,ctypes.byref(bufI))
        iEst(inImg,0,i32E(0),i32E(W))
        iEst(inImg,1,i32E(0),i32E(H))
        
        # defs
        orig     = C.hwrap_img_to_func(inImg)
        x        = C.hwrap_new_var(b"x")
        y        = C.hwrap_new_var(b"y")
        
        blur_x, blur_y = self.build_blur(orig,x,y)
        fEst(blur_y,0,i32E(1),i32E(W-2))
        fEst(blur_y,1,i32E(1),i32E(H-2))

        if use_auto:
            print("Using auto-scheduler")
            pipe = C.hwrap_new_pipeline(1,(1 * hw_func_t)(blur_y))

            # run tests
            t0       = time.perf_counter()
            C.hwrap_realize_pipeline(pipe, 1, (1*halide_buffer_t)(bufO))
            t1       = time.perf_counter()
            C.hwrap_realize_pipeline(pipe, 1, (1*halide_buffer_t)(bufO))
            t2       = time.perf_counter()
            for k in range(0,10):
                C.hwrap_realize_pipeline(pipe, 1, (1*halide_buffer_t)(bufO))
            t3       = time.perf_counter()
            print('blur times', t1-t0,t2-t1,t3-t2)

        else:
            # run tests
            t0       = time.perf_counter()
            C.hwrap_realize_func(blur_y,bufO)
            t1       = time.perf_counter()
            C.hwrap_realize_func(blur_y,bufO)
            t2       = time.perf_counter()
            for k in range(0,10):
                C.hwrap_realize_func(blur_y,bufO)
            t3       = time.perf_counter()
            print('blur times', t1-t0,t2-t1,t3-t2)

    def test_blur_0(self):
        print()
        self.blur_test_0()
    
    def test_blur_0_autoschedule(self):
        print()
        self.blur_test_0(True)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()
