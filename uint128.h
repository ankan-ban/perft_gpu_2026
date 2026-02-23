#pragma once

#include "chess.h"
#include <string.h>

// Simple 128-bit unsigned integer for CPU-side accumulation at deep perft levels
struct uint128
{
    uint64 lo;
    uint64 hi;

    uint128() : lo(0), hi(0) {}
    uint128(uint64 v) : lo(v), hi(0) {}

    uint128 &operator+=(uint64 v)
    {
        uint64 old = lo;
        lo += v;
        if (lo < old) hi++;
        return *this;
    }

    uint128 &operator+=(const uint128 &other)
    {
        uint64 old = lo;
        lo += other.lo;
        uint64 carry = (lo < old) ? 1 : 0;
        hi += other.hi + carry;
        return *this;
    }

    uint128 operator+(const uint128 &other) const
    {
        uint128 result = *this;
        result += other;
        return result;
    }

    bool operator==(const uint128 &other) const { return lo == other.lo && hi == other.hi; }
    bool operator!=(const uint128 &other) const { return !(*this == other); }

    double toDouble() const { return (double)hi * 18446744073709551616.0 + (double)lo; }

    // Convert to decimal string for printing
    // Uses repeated division by 10^9 for efficiency
    void toDecimalString(char *buf, int bufSize) const
    {
        if (hi == 0)
        {
            snprintf(buf, bufSize, "%llu", (unsigned long long)lo);
            return;
        }

        // Divide 128-bit number by 10^9 repeatedly to extract groups of 9 digits
        // Working with 32-bit chunks for simpler division
        uint32 parts[4] = {
            (uint32)(lo & 0xFFFFFFFF),
            (uint32)(lo >> 32),
            (uint32)(hi & 0xFFFFFFFF),
            (uint32)(hi >> 32)
        };

        char groups[5][10];  // max 5 groups of 9 digits for 128-bit
        int nGroups = 0;
        const uint64 divisor = 1000000000ULL;

        while (parts[3] || parts[2] || parts[1] || parts[0])
        {
            // Divide the 128-bit number (in 4 x 32-bit parts, big-endian order) by 10^9
            uint64 remainder = 0;
            for (int i = 3; i >= 0; i--)
            {
                uint64 cur = (remainder << 32) | parts[i];
                parts[i] = (uint32)(cur / divisor);
                remainder = cur % divisor;
            }
            snprintf(groups[nGroups++], 10, "%09u", (uint32)remainder);
        }

        if (nGroups == 0)
        {
            snprintf(buf, bufSize, "0");
            return;
        }

        // First group: strip leading zeros
        char *p = buf;
        int firstLen = (int)strlen(groups[nGroups - 1]);
        char *firstStart = groups[nGroups - 1];
        while (*firstStart == '0' && firstStart[1] != '\0') firstStart++;
        int trimmedLen = (int)strlen(firstStart);
        if (p + trimmedLen < buf + bufSize - 1)
        {
            memcpy(p, firstStart, trimmedLen);
            p += trimmedLen;
        }

        // Remaining groups: print all 9 digits
        for (int g = nGroups - 2; g >= 0; g--)
        {
            if (p + 9 < buf + bufSize - 1)
            {
                memcpy(p, groups[g], 9);
                p += 9;
            }
        }
        *p = '\0';
    }
};
