#ifndef PRECISION_H
#define PRECISION_H

struct int4_t {
    __host__ __device__ int4_t(int value) {
        // truncate to within valid int4 range: -8 to 7
        if (value > 7)
            value_ = 7;
        else if (value < -8)
            value_ = -8;
        else
            value_ = value;
    };

    operator int() {
        return value_;
    }
    private:
        int value_;
};

#endif // PRECISION_H