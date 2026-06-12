// SPDX-License-Identifier: Apache-2.0
//
// Tiny fixed-size matrices that pair with the Vec types in vec.cuh.
//
// Conventions
// -----------
// * Storage is row-major.  The element at row r, column c lives at
//   `data[r * cols + c]` and is accessed via `m(r, c)`.
// * Mutating ops (`+=`, `-=`, `*= scalar`) live on the class.  All
//   binary ops, transposition, determinants and inverses are
//   non-members and never modify their arguments.
// * `inverse(m)` always returns a fresh matrix.  The caller is
//   responsible for guaranteeing the input is non-singular; this is a
//   minimal library on purpose, so no graceful pseudo-inverse fallback
//   is provided.
// * `outer(a, b)` produces the rank-1 matrix `a b^T`.

#pragma once

#include "common.cuh"
#include "vec.cuh"

namespace chysx {
namespace math {

// ============================================================================
// Mat2
// ============================================================================

template <typename T>
struct Mat2 {
    // row-major: data[r*2 + c]
    T data[4];

    CHYSX_HD Mat2() : data{T{}, T{}, T{}, T{}} {}
    CHYSX_HD Mat2(T m00, T m01,
                  T m10, T m11)
        : data{m00, m01,
               m10, m11} {}

    CHYSX_HD T& operator()(int r, int c)             { return data[r * 2 + c]; }
    CHYSX_HD const T& operator()(int r, int c) const { return data[r * 2 + c]; }

    CHYSX_HD Mat2& operator+=(const Mat2& o) {
        for (int i = 0; i < 4; ++i) data[i] += o.data[i];
        return *this;
    }
    CHYSX_HD Mat2& operator-=(const Mat2& o) {
        for (int i = 0; i < 4; ++i) data[i] -= o.data[i];
        return *this;
    }
    CHYSX_HD Mat2& operator*=(T s) {
        for (int i = 0; i < 4; ++i) data[i] *= s;
        return *this;
    }

    static CHYSX_HD Mat2 identity() {
        return Mat2(T{1}, T{0},
                    T{0}, T{1});
    }
    static CHYSX_HD Mat2 zero() { return Mat2(); }
};

template <typename T>
CHYSX_HDI Mat2<T> operator+(Mat2<T> a, const Mat2<T>& b) { return a += b; }
template <typename T>
CHYSX_HDI Mat2<T> operator-(Mat2<T> a, const Mat2<T>& b) { return a -= b; }
template <typename T>
CHYSX_HDI Mat2<T> operator*(Mat2<T> a, T s)              { return a *= s; }
template <typename T>
CHYSX_HDI Mat2<T> operator*(T s, Mat2<T> a)              { return a *= s; }

template <typename T>
CHYSX_HDI Mat2<T> operator-(const Mat2<T>& a) {
    return Mat2<T>(-a(0,0), -a(0,1),
                   -a(1,0), -a(1,1));
}

template <typename T>
CHYSX_HDI Mat2<T> operator*(const Mat2<T>& a, const Mat2<T>& b) {
    Mat2<T> r;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            r(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j);
        }
    }
    return r;
}

template <typename T>
CHYSX_HDI Vec2<T> operator*(const Mat2<T>& m, const Vec2<T>& v) {
    return Vec2<T>(m(0, 0) * v.x + m(0, 1) * v.y,
                   m(1, 0) * v.x + m(1, 1) * v.y);
}

template <typename T>
CHYSX_HDI Mat2<T> transpose(const Mat2<T>& m) {
    return Mat2<T>(m(0, 0), m(1, 0),
                   m(0, 1), m(1, 1));
}

template <typename T>
CHYSX_HDI T trace(const Mat2<T>& m) { return m(0, 0) + m(1, 1); }

template <typename T>
CHYSX_HDI T determinant(const Mat2<T>& m) {
    return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}

template <typename T>
CHYSX_HDI Mat2<T> inverse(const Mat2<T>& m) {
    T inv_det = T{1} / determinant(m);
    return Mat2<T>( m(1, 1) * inv_det, -m(0, 1) * inv_det,
                   -m(1, 0) * inv_det,  m(0, 0) * inv_det);
}

template <typename T>
CHYSX_HDI Mat2<T> outer(const Vec2<T>& a, const Vec2<T>& b) {
    return Mat2<T>(a.x * b.x, a.x * b.y,
                   a.y * b.x, a.y * b.y);
}

// ============================================================================
// Mat3
// ============================================================================

template <typename T>
struct Mat3 {
    // row-major: data[r*3 + c]
    T data[9];

    CHYSX_HD Mat3() : data{} {}
    CHYSX_HD Mat3(T m00, T m01, T m02,
                  T m10, T m11, T m12,
                  T m20, T m21, T m22)
        : data{m00, m01, m02,
               m10, m11, m12,
               m20, m21, m22} {}

    CHYSX_HD T& operator()(int r, int c)             { return data[r * 3 + c]; }
    CHYSX_HD const T& operator()(int r, int c) const { return data[r * 3 + c]; }

    CHYSX_HD Mat3& operator+=(const Mat3& o) {
        for (int i = 0; i < 9; ++i) data[i] += o.data[i];
        return *this;
    }
    CHYSX_HD Mat3& operator-=(const Mat3& o) {
        for (int i = 0; i < 9; ++i) data[i] -= o.data[i];
        return *this;
    }
    CHYSX_HD Mat3& operator*=(T s) {
        for (int i = 0; i < 9; ++i) data[i] *= s;
        return *this;
    }

    static CHYSX_HD Mat3 identity() {
        return Mat3(T{1}, T{0}, T{0},
                    T{0}, T{1}, T{0},
                    T{0}, T{0}, T{1});
    }
    static CHYSX_HD Mat3 zero() { return Mat3(); }
};

template <typename T>
CHYSX_HDI Mat3<T> operator+(Mat3<T> a, const Mat3<T>& b) { return a += b; }
template <typename T>
CHYSX_HDI Mat3<T> operator-(Mat3<T> a, const Mat3<T>& b) { return a -= b; }
template <typename T>
CHYSX_HDI Mat3<T> operator*(Mat3<T> a, T s)              { return a *= s; }
template <typename T>
CHYSX_HDI Mat3<T> operator*(T s, Mat3<T> a)              { return a *= s; }

template <typename T>
CHYSX_HDI Mat3<T> operator-(const Mat3<T>& a) {
    Mat3<T> r;
    for (int i = 0; i < 9; ++i) r.data[i] = -a.data[i];
    return r;
}

template <typename T>
CHYSX_HDI Mat3<T> operator*(const Mat3<T>& a, const Mat3<T>& b) {
    Mat3<T> r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r(i, j) = a(i, 0) * b(0, j)
                    + a(i, 1) * b(1, j)
                    + a(i, 2) * b(2, j);
        }
    }
    return r;
}

template <typename T>
CHYSX_HDI Vec3<T> operator*(const Mat3<T>& m, const Vec3<T>& v) {
    return Vec3<T>(m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z,
                   m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z,
                   m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z);
}

template <typename T>
CHYSX_HDI Mat3<T> transpose(const Mat3<T>& m) {
    return Mat3<T>(m(0, 0), m(1, 0), m(2, 0),
                   m(0, 1), m(1, 1), m(2, 1),
                   m(0, 2), m(1, 2), m(2, 2));
}

template <typename T>
CHYSX_HDI T trace(const Mat3<T>& m) {
    return m(0, 0) + m(1, 1) + m(2, 2);
}

template <typename T>
CHYSX_HDI T determinant(const Mat3<T>& m) {
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
         - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
         + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

template <typename T>
CHYSX_HDI Mat3<T> inverse(const Mat3<T>& m) {
    // adj(M) / det(M) — closed-form, no destructive Gauss-Jordan.
    Mat3<T> a;
    a(0, 0) =  m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    a(0, 1) = -m(0, 1) * m(2, 2) + m(0, 2) * m(2, 1);
    a(0, 2) =  m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
    a(1, 0) = -m(1, 0) * m(2, 2) + m(1, 2) * m(2, 0);
    a(1, 1) =  m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
    a(1, 2) = -m(0, 0) * m(1, 2) + m(0, 2) * m(1, 0);
    a(2, 0) =  m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    a(2, 1) = -m(0, 0) * m(2, 1) + m(0, 1) * m(2, 0);
    a(2, 2) =  m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

    T det = m(0, 0) * a(0, 0) + m(0, 1) * a(1, 0) + m(0, 2) * a(2, 0);
    return a * (T{1} / det);
}

template <typename T>
CHYSX_HDI Mat3<T> outer(const Vec3<T>& a, const Vec3<T>& b) {
    return Mat3<T>(a.x * b.x, a.x * b.y, a.x * b.z,
                   a.y * b.x, a.y * b.y, a.y * b.z,
                   a.z * b.x, a.z * b.y, a.z * b.z);
}

// ============================================================================
// Mat4
// ============================================================================

template <typename T>
struct Mat4 {
    // row-major: data[r*4 + c]
    T data[16];

    CHYSX_HD Mat4() : data{} {}
    CHYSX_HD Mat4(T m00, T m01, T m02, T m03,
                  T m10, T m11, T m12, T m13,
                  T m20, T m21, T m22, T m23,
                  T m30, T m31, T m32, T m33)
        : data{m00, m01, m02, m03,
               m10, m11, m12, m13,
               m20, m21, m22, m23,
               m30, m31, m32, m33} {}

    CHYSX_HD T& operator()(int r, int c)             { return data[r * 4 + c]; }
    CHYSX_HD const T& operator()(int r, int c) const { return data[r * 4 + c]; }

    CHYSX_HD Mat4& operator+=(const Mat4& o) {
        for (int i = 0; i < 16; ++i) data[i] += o.data[i];
        return *this;
    }
    CHYSX_HD Mat4& operator-=(const Mat4& o) {
        for (int i = 0; i < 16; ++i) data[i] -= o.data[i];
        return *this;
    }
    CHYSX_HD Mat4& operator*=(T s) {
        for (int i = 0; i < 16; ++i) data[i] *= s;
        return *this;
    }

    static CHYSX_HD Mat4 identity() {
        return Mat4(T{1}, T{0}, T{0}, T{0},
                    T{0}, T{1}, T{0}, T{0},
                    T{0}, T{0}, T{1}, T{0},
                    T{0}, T{0}, T{0}, T{1});
    }
    static CHYSX_HD Mat4 zero() { return Mat4(); }
};

template <typename T>
CHYSX_HDI Mat4<T> operator+(Mat4<T> a, const Mat4<T>& b) { return a += b; }
template <typename T>
CHYSX_HDI Mat4<T> operator-(Mat4<T> a, const Mat4<T>& b) { return a -= b; }
template <typename T>
CHYSX_HDI Mat4<T> operator*(Mat4<T> a, T s)              { return a *= s; }
template <typename T>
CHYSX_HDI Mat4<T> operator*(T s, Mat4<T> a)              { return a *= s; }

template <typename T>
CHYSX_HDI Mat4<T> operator-(const Mat4<T>& a) {
    Mat4<T> r;
    for (int i = 0; i < 16; ++i) r.data[i] = -a.data[i];
    return r;
}

template <typename T>
CHYSX_HDI Mat4<T> operator*(const Mat4<T>& a, const Mat4<T>& b) {
    Mat4<T> r;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            r(i, j) = a(i, 0) * b(0, j)
                    + a(i, 1) * b(1, j)
                    + a(i, 2) * b(2, j)
                    + a(i, 3) * b(3, j);
        }
    }
    return r;
}

template <typename T>
CHYSX_HDI Vec4<T> operator*(const Mat4<T>& m, const Vec4<T>& v) {
    return Vec4<T>(m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z + m(0, 3) * v.w,
                   m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z + m(1, 3) * v.w,
                   m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z + m(2, 3) * v.w,
                   m(3, 0) * v.x + m(3, 1) * v.y + m(3, 2) * v.z + m(3, 3) * v.w);
}

template <typename T>
CHYSX_HDI Mat4<T> transpose(const Mat4<T>& m) {
    Mat4<T> r;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            r(i, j) = m(j, i);
        }
    }
    return r;
}

template <typename T>
CHYSX_HDI T trace(const Mat4<T>& m) {
    return m(0, 0) + m(1, 1) + m(2, 2) + m(3, 3);
}

namespace detail {

// Determinant of the 3x3 minor of `m` formed by deleting row `skip_r` and
// column `skip_c`.  Helper used by both determinant() and inverse().
template <typename T>
CHYSX_HDI T mat4_minor3(const Mat4<T>& m, int skip_r, int skip_c) {
    T sub[9];
    int idx = 0;
    for (int r = 0; r < 4; ++r) {
        if (r == skip_r) continue;
        for (int c = 0; c < 4; ++c) {
            if (c == skip_c) continue;
            sub[idx++] = m(r, c);
        }
    }
    return sub[0] * (sub[4] * sub[8] - sub[5] * sub[7])
         - sub[1] * (sub[3] * sub[8] - sub[5] * sub[6])
         + sub[2] * (sub[3] * sub[7] - sub[4] * sub[6]);
}

}  // namespace detail

template <typename T>
CHYSX_HDI T determinant(const Mat4<T>& m) {
    // Cofactor expansion along the first row.
    T c0 =  detail::mat4_minor3(m, 0, 0);
    T c1 = -detail::mat4_minor3(m, 0, 1);
    T c2 =  detail::mat4_minor3(m, 0, 2);
    T c3 = -detail::mat4_minor3(m, 0, 3);
    return m(0, 0) * c0 + m(0, 1) * c1 + m(0, 2) * c2 + m(0, 3) * c3;
}

template <typename T>
CHYSX_HDI Mat4<T> inverse(const Mat4<T>& m) {
    // Cofactor matrix; inverse = (cofactor)^T / det.  We pay the cost of
    // recomputing minors here for clarity — the loop is short enough that
    // nvcc inlines it cleanly, and Mat4 inversion isn't on the hot path
    // for a particle simulator.
    Mat4<T> cof;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            T sgn = ((r + c) & 1) ? T{-1} : T{1};
            cof(r, c) = sgn * detail::mat4_minor3(m, r, c);
        }
    }

    T det = m(0, 0) * cof(0, 0) + m(0, 1) * cof(0, 1)
          + m(0, 2) * cof(0, 2) + m(0, 3) * cof(0, 3);
    T inv_det = T{1} / det;

    Mat4<T> r;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            r(i, j) = cof(j, i) * inv_det;
        }
    }
    return r;
}

template <typename T>
CHYSX_HDI Mat4<T> outer(const Vec4<T>& a, const Vec4<T>& b) {
    return Mat4<T>(a.x * b.x, a.x * b.y, a.x * b.z, a.x * b.w,
                   a.y * b.x, a.y * b.y, a.y * b.z, a.y * b.w,
                   a.z * b.x, a.z * b.y, a.z * b.z, a.z * b.w,
                   a.w * b.x, a.w * b.y, a.w * b.z, a.w * b.w);
}

// ============================================================================
// Type aliases for named matrices
// ============================================================================

using Mat2f = Mat2<float>;
using Mat3f = Mat3<float>;
using Mat4f = Mat4<float>;

using Mat2d = Mat2<double>;
using Mat3d = Mat3<double>;
using Mat4d = Mat4<double>;

// ============================================================================
// VecN<T, N> — generic fixed-size vector
// ============================================================================

template <typename T, int N>
struct VecN {
    T data[N];

    CHYSX_HD T& operator[](int i) { return data[i]; }
    CHYSX_HD const T& operator[](int i) const { return data[i]; }

    CHYSX_HD VecN& operator+=(const VecN& o) {
        for (int i = 0; i < N; ++i) data[i] += o.data[i]; return *this;
    }
    CHYSX_HD VecN& operator-=(const VecN& o) {
        for (int i = 0; i < N; ++i) data[i] -= o.data[i]; return *this;
    }
    CHYSX_HD VecN& operator*=(T s) {
        for (int i = 0; i < N; ++i) data[i] *= s; return *this;
    }

    static CHYSX_HD VecN zero() { VecN v; for (int i = 0; i < N; ++i) v.data[i] = T{0}; return v; }
};

template <typename T, int N>
CHYSX_HDI VecN<T, N> operator+(VecN<T, N> a, const VecN<T, N>& b) { return a += b; }
template <typename T, int N>
CHYSX_HDI VecN<T, N> operator-(VecN<T, N> a, const VecN<T, N>& b) { return a -= b; }
template <typename T, int N>
CHYSX_HDI VecN<T, N> operator*(VecN<T, N> a, T s) { return a *= s; }
template <typename T, int N>
CHYSX_HDI VecN<T, N> operator*(T s, VecN<T, N> a) { return a *= s; }
template <typename T, int N>
CHYSX_HDI VecN<T, N> operator-(const VecN<T, N>& a) {
    VecN<T, N> r; for (int i = 0; i < N; ++i) r[i] = -a[i]; return r;
}

template <typename T, int N>
CHYSX_HDI T dot(const VecN<T, N>& a, const VecN<T, N>& b) {
    T s = T{0}; for (int i = 0; i < N; ++i) s += a[i] * b[i]; return s;
}

template <typename T, int N>
CHYSX_HDI T max_abs(const VecN<T, N>& v) {
    T m = T{0};
    for (int i = 0; i < N; ++i) { T a = abs(v[i]); if (a > m) m = a; }
    return m;
}

// Convenience: embed / extract Vec3 in a VecN
template <typename T, int N>
CHYSX_HDI void set_block3(VecN<T, N>& v, int offset, Vec3<T> b) {
    v[offset] = b.x; v[offset + 1] = b.y; v[offset + 2] = b.z;
}

template <typename T, int N>
CHYSX_HDI Vec3<T> get_block3(const VecN<T, N>& v, int offset) {
    return Vec3<T>(v[offset], v[offset + 1], v[offset + 2]);
}

// ============================================================================
// MatN<T, N> — generic fixed-size NxN square matrix (row-major)
// ============================================================================

template <typename T, int N>
struct MatN {
    T data[N * N];

    CHYSX_HD T& operator()(int r, int c) { return data[r * N + c]; }
    CHYSX_HD const T& operator()(int r, int c) const { return data[r * N + c]; }

    CHYSX_HD MatN& operator+=(const MatN& o) {
        for (int i = 0; i < N * N; ++i) data[i] += o.data[i]; return *this;
    }
    CHYSX_HD MatN& operator-=(const MatN& o) {
        for (int i = 0; i < N * N; ++i) data[i] -= o.data[i]; return *this;
    }
    CHYSX_HD MatN& operator*=(T s) {
        for (int i = 0; i < N * N; ++i) data[i] *= s; return *this;
    }

    static CHYSX_HD MatN zero() {
        MatN m; for (int i = 0; i < N * N; ++i) m.data[i] = T{0}; return m;
    }
    static CHYSX_HD MatN identity() {
        MatN m = zero();
        for (int i = 0; i < N; ++i) m(i, i) = T{1};
        return m;
    }
};

template <typename T, int N>
CHYSX_HDI MatN<T, N> operator+(MatN<T, N> a, const MatN<T, N>& b) { return a += b; }
template <typename T, int N>
CHYSX_HDI MatN<T, N> operator-(MatN<T, N> a, const MatN<T, N>& b) { return a -= b; }
template <typename T, int N>
CHYSX_HDI MatN<T, N> operator*(MatN<T, N> a, T s) { return a *= s; }
template <typename T, int N>
CHYSX_HDI MatN<T, N> operator*(T s, MatN<T, N> a) { return a *= s; }

template <typename T, int N>
CHYSX_HDI VecN<T, N> operator*(const MatN<T, N>& M, const VecN<T, N>& v) {
    VecN<T, N> r = VecN<T, N>::zero();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r[i] += M(i, j) * v[j];
    return r;
}

template <typename T, int N>
CHYSX_HDI MatN<T, N> transpose(const MatN<T, N>& m) {
    MatN<T, N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r(i, j) = m(j, i);
    return r;
}

// Embed/extract 3x3 block inside an NxN matrix
template <typename T, int N>
CHYSX_HDI void set_block3x3(MatN<T, N>& M, int row, int col, const Mat3<T>& B) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            M(row + r, col + c) = B(r, c);
}

template <typename T, int N>
CHYSX_HDI Mat3<T> get_block3x3(const MatN<T, N>& M, int row, int col) {
    Mat3<T> B;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            B(r, c) = M(row + r, col + c);
    return B;
}

// ============================================================================
// Cholesky factorization / solve for SPD MatN
// ============================================================================

// In-place LL^T factorization. Writes the lower triangle of L.
// Clamps near-zero pivots for robustness on GPU.
template <typename T, int N>
CHYSX_HDI void cholesky(MatN<T, N>& L, const MatN<T, N>& M) {
    for (int i = 0; i < N * N; ++i) L.data[i] = M.data[i];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            T s = L(i, j);
            for (int k = 0; k < j; ++k) s -= L(i, k) * L(j, k);
            if (i == j) {
                if (s <= T(1e-20)) s = T(1e-20);
                L(i, j) = sqrt(s);
            } else {
                L(i, j) = s / L(j, j);
            }
        }
        for (int j = i + 1; j < N; ++j) L(i, j) = T{0};
    }
}

// Solve L L^T x = b given lower-triangular Cholesky factor L.
template <typename T, int N>
CHYSX_HDI VecN<T, N> cholesky_solve(const MatN<T, N>& L, const VecN<T, N>& b) {
    VecN<T, N> y = VecN<T, N>::zero();
    for (int i = 0; i < N; ++i) {
        T s = b[i];
        for (int k = 0; k < i; ++k) s -= L(i, k) * y[k];
        y[i] = s / L(i, i);
    }
    VecN<T, N> x = VecN<T, N>::zero();
    for (int i = N - 1; i >= 0; --i) {
        T s = y[i];
        for (int k = i + 1; k < N; ++k) s -= L(k, i) * x[k];
        x[i] = s / L(i, i);
    }
    return x;
}

// SPD inverse via Cholesky: M^{-1} by solving M x_i = e_i for each column.
template <typename T, int N>
CHYSX_HDI MatN<T, N> inverse_spd(const MatN<T, N>& M) {
    MatN<T, N> L;
    cholesky<T, N>(L, M);
    MatN<T, N> inv = MatN<T, N>::zero();
    for (int col = 0; col < N; ++col) {
        VecN<T, N> e = VecN<T, N>::zero();
        e[col] = T{1};
        VecN<T, N> x = cholesky_solve<T, N>(L, e);
        for (int row = 0; row < N; ++row) inv(row, col) = x[row];
    }
    return inv;
}

// Diagonal regularization: ensure all diagonal entries >= eps.
// Quick positive-semidefinite fix for small dense Hessians on GPU.
template <typename T, int N>
CHYSX_HDI void make_spd(MatN<T, N>& H) {
    // Symmetrize
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            H(i, j) = H(j, i) = T(0.5) * (H(i, j) + H(j, i));
    // Regularize diagonals
    T tr = T{0};
    for (int i = 0; i < N; ++i) tr += abs(H(i, i));
    T eps = tr * T(1e-6);
    if (eps < T(1e-8)) eps = T(1e-8);
    for (int i = 0; i < N; ++i)
        if (H(i, i) < eps) H(i, i) = eps;
}

// ============================================================================
// Type aliases for VecN / MatN
// ============================================================================

using Vec9f  = VecN<float, 9>;
using Vec12f = VecN<float, 12>;
using Mat9f  = MatN<float, 9>;
using Mat12f = MatN<float, 12>;

using Vec9d  = VecN<double, 9>;
using Vec12d = VecN<double, 12>;
using Mat9d  = MatN<double, 9>;
using Mat12d = MatN<double, 12>;

}  // namespace math
}  // namespace chysx
