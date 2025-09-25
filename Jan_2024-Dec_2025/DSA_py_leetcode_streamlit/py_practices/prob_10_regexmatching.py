"""
    Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

    '.' Matches any single character
    '*' Matches zero or more of the preceding element.
    The matching should cover the entire input string (not partial).
"""

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        len_s, len_p = len(s), len(p)
        dp = [[False] * (len_p + 1) for _ in range(len_s + 1)]
        dp[0][0] = True  # empty string matches empty pattern

        # handle patterns like a*, a*b*, a*b*c*
        for j in range(2, len_p + 1):
            print(s[j-1], p[j-1])
            if p[j-1] == "*":
                print(s[j-1], p[j-1])
                dp[0][j] = dp[0][j-2]

        for i in range(1, len_s + 1):
            for j in range(1, len_p + 1):
                if p[j-1] == "." or p[j-1] == s[i-1]:
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == "*":
                    dp[i][j] = dp[i][j-2] or (
                        dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2] == ".")
                    )

        print(dp, len_s, len_p)
        return dp[len_s][len_p]
    
s = "aaaa"
p = "a*a"
sol = Solution()
print(sol.isMatch(s, p))
