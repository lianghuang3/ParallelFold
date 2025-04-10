// g++ --std=c++11 rna.cpp -o rna
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

// Function to count RNA secondary structures using the KT formulation.
long long total(const std::string &s) {
    int n = s.size();

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // Base cases: dp[i][i] = 1 (empty subsequence)
    // and dp[i][i+1] = 1 (singleton subsequence).
    for (int i = 0; i <= n; ++i) {
        dp[i][i] = 1;
        if(i < n) {
            dp[i][i + 1] = 1;
        }
    }

    // Fill the DP table.
    // span is the length of the subsequence we are considering.
    for (int span = 2; span <= n; ++span) {
        for (int i = 0; i <= n - span; ++i) {
            int j = i + span;
            // Case 1: The base at i is unpaired.
            dp[i][j] += dp[i + 1][j];

            // Case 2: Try to pair base at i with a base at position k-1.
            // k goes from i+1 to j (k-1 from i to j-1).
            for (int k = i + 1; k <= j; ++k) {
                // Form the pair: s[i] and s[k - 1]
                std::string pair = { s[i], s[k - 1] };
                if (allowed.count(pair)) {
                    dp[i][j] += dp[i + 1][k - 1] * dp[k][j];
                }
            }
        }
    }
    return dp[0][n];
}

int main() {
    // Test cases: 16S rRNA
    std::string test1 = "AUUCUGGUUGAUCCUGCCAGAGGCCGCUGCUAUCCGGCUGGGACUAAGCCAUGCGAGUCAAGGGGCUUGUAUCCCUUCGGGGAUGCAAGCACCGGCGGACGGCUCAGUAACACGUGGACAACCUGCCCUCGGGUGGGGGAUAACCCCGGGAAACUGGGGCUAAUCCCCCAUAGGGGAUGGGUACUGGAAUGUCCCAUCUCCGAAAGCGCUUAGCGCCCGAGGAUGGGUCUGCGGCGGAUUAGGUUGUUGGUGGGGUAACGGCCCACCAAGCCGAAGAUCCGUACGGGCCAUGAGAGUGGGAGCCCGGAGAUGGACCCUGAGACACGGGUCCAGGCCCUACGGGGCGCAGCAGGCGCGAAACCUCCGCAAUGCGGGAAACCGCGACGGGGUCAGCCGGAGUGCUCGCGCAUCGCGCGGGCUGUCGGGGUGCCUAAAAAGCACCCCACAGCAAGGGCCGGGCAAGGCCGGUGGCAGCCGCCGCGGUAAUACCGGCGGCCCGAGUGGCGGCCACUUUUAUUGGGCCUAAAGCGUCCGUAGCCGGGCUGGUAAGUCCUCCGGGAAAUCUGGCGGCUUAACCGUCAGACUGCCGGAGGAUACUGCCAGCCUAGGGACCGGGAGAGGCCGGGGGUAUUCCCGGAGUAGGGGUGAAAUCCUGUAAUCCCGGGAGGACCACCUGUGGCGAAGGCGCCCGGCUGGAACGGGUCCGACGGUGAGGGACGAAGGCCAGGGGAGCGAACCGGAUUAGAUACCCGGGUAGUCCUGGCUGUAAACGAUGCGGACUAGGUGUCACCGAAGCUACGAGCUUCGGUGGUGCCGGAGGGAAGCCGUUAAGUCCGCCGCCUGGGGAGUACGGCCGCAAGGCUGAAACUUAAAGGAAUUGGCGGGGGAGCACUACAACGGGUGGAGCCUGCGGUUUAAUUGGAUUCAACGCCGGGAAGCUUACCGGGGGAGACAGCGGGAUGAAGGUCGGGCUGAAGACCUUACCAGACUAGCUGAGAGGUGGUGCAUGGCCGCCGUCAGUUCGUACUGUGAAGCAUCCUGUUAAGUCAGGCAACGAGCGAGACCCGCGCCCCCAGUUGCCAGCGGUUCCCUUCGGGGAAGCCGGGCACACUGGGGGGACUGCCGGCGCUAAGCCGGAGGAAGGUGCGGGCAACGGCAGGUCCGUAUGCCCCGAAUCCCCCGGGCUACACGCGGGCUACAAUGGCCGGGACAAUGGGUACCGACCCCGAAAGGGGUAGGUAAUCCCCUAAACCCGGUCUAACCUGGGAUCGAGGGCUGCAACUCGCCCUCGUGAACCUGGAAUCCGUAGUAAUCGCGCCUCAAAAUGGCGCGGUGAAUACGUCCCUGCUCCUUGCACACACCGCCCGUCAAGCCACCCGAGUGGGCCAGGGGCGAGGGGGUGGCCCUAGGCCACCUUCGAGCCCAGGGUCCGCGAGGGGGGCUAAGUCGUAACAAGGUAGCCGUAGGGGAAUCUGCGGCUGGAUCACCUCCU";
    std::string test2 = "UUCCCUGAAGAGUUUGAUCCUGGCUCAGCGCGAACGCUGGCGGCGUGCCUAACACAUGCAAGUCGUGCGCAGGCUCGCUCCCUCUGGGAGCGGGUGCUGAGCGGCAAACGGGUGAGUAACACGUGGGUAACCUACCCCCAGGAGGGGGAUAACCCCGGGAAACCGGGGCUAAUACCCCAUAAAGCCGCCCGCCACUAAGGCGAGGCGGCCAAAGGGGGCCUCUGGGCUCUGCCCAAGCUCCCGCCUGGGGAUGGGCCCGCGGCCCAUCAGGUAGUUGGUGGGGUAACGGCCCACCAAGCCUAUGACGGGUAGCCGGCCUGAGAGGGUGGCCGGCCACAGCGGGACUGAGACACGGCCCGCACCCCUACGGGGGGCAGCAGUGGGGAAUCGUGGGCAAUGGGCGAAAGCCUGACCCCGCGACGCCGCGUGGGGGAAGAAGCCCUGCGGGGUGUAAACCCCUGUCGGGGGGGACGAAGGGACUGUGGGUUAAUAGCCCACAGUCUUGACGGUACCCCCAGAGGAAGGGACGGCUAACUACGUGCCAGCAGCCGCGGUAAUACGUAGGUCCCGAGCGUUGCGCGAAGUCACUGGGCGUAAAGCGUCCGCAGCCGGUCGGGUAAGCGGGAUGUCAAAGCCCACGGCUCAACCGUGGAAUGGCAUCCCGAACUGCCCGACUUGAGGCACGCCCGGGCAGGCGGAAUUCCCGGGGUAGCGGUGAAAUGCGUAGAUCUCGGGAGGAACACCGAAGGGGAAGCCAGCCUGCUGGGGCUGUCCUGACGGUCAGGGACGAAAGCCGGGGGAGCGAACCGGAUUAGAUACCCGGGUAGUCCCGGCCGUAAACCAUGGGCGCUAGGGCUUGUCCCUUUGGGGCAGGCUCGCAGCUAACGCGUUAAGCGCCCCGCCUGGGGAGUACGGGCGCAAGCCUGAAACUCAAAGGAAUUGGCGGGGGCCCGCACAACCGGUGGAGCGUCUGGUUCAAUUCGAUGCUAACCGAAGAACCUUACCCGGGCUUGACAUGCCGGGGAGACUCCGCGAAAGCGGAGUUGUGGAAGUCUCUGACUUCCCCCCGGCACAGGUGGUGCAUGGCCGUCGUCAGCUCGUGUCGUGAGAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCCUGCCCCUAGUUGCUACCCCGAGAGGGGAGCACUCUAGGGGGACCGCCGGCGAUAAGCCGGAGGAAGGGGGGGAUGACGUCAGGUCAGUAUGCCCUUUAUGCCCGGGGCCACACAGGCGCUACAGUGGCCGGGACAAUGGGAAGCGACCCCGCAAGGGGGAGCUAAUCCCAGAAACCCGGUCAUGGUGCGGAUUGGGGGCUGAAACUCGCCCCCAUGAAGCCGGAAUCGGUAGUAACGGGGUAUCAGCGAUGUCCCCGUGAAUACGUUCUCGGGCCUUGCACACACCGCCCGUCACGCCACGGAAGUCGGUCCGGCCGGAAGUCCCCGAGCUAACCGGCCCUUUUUGGGCCGGGGGCAGGGGCCGAUGGCCGGGCCGGCGACUGGGGCGAAGUCGUAACAAGGUAGCCGUAGGGGAACCUGC";
    // Test Cases: GRP II 
    std::string test3 = "AGUUUAGUGGUAAAAGUGUGAUUCGUUCUAUUAUCCCUUAAAUAGUUAAAGGGUCCUUCGGUUUGAUUCGUAUUCCGAUCAAAAACUUGAUUUCUAAAAAGGAUUUAAUCCUUUUCCUCUCAAUGACAGAUUCGAGAACAAAUACACAUUCUCGUGAUUUGUAUCCAAGGGUCACUUAGACAUUGAAAAAUUGGAUUAUGAAAUUGCGAAACAUAAUUUUGGAAUUGGAUCAAUACUUCCAAUUGAAUAAGUAUGAAUAAAGGAUCCAUGGAUGAAGAUAGAAAGUUGAUUUCUAAUCGUAACUAAAUCUUCAAUUUCUUAUUUGUAAAGAAGAAAUUGAAGCAAAAUAGCUAUUAAACGAUGACUUUGGUUUACUAGAGACAUCAACAUAUUGUUUUAGCUCGGUGGAAACAAAACCCUUUUCCUCAGGAUCCUAUUAAAUAGAAAUAGAGAACGAAAUAACUAGAAAGGUUGUUAGAAUCCCCUCUUCUAGAAGGAUCAUCUACAAAGCUAUUCGUUUUAUCUGUAUUCAGACCAAAAGCUGACAUAGAUGUUAUGGGUAGAAUUCUUUUUUUUUUUCGAAUUUUGUUCACAUCUUAGAUCUAUAAAUUGACUCAUCUCCAUAAAGGAGCCGAAUGAAACCAAAGUUUCAUGUUCGGUUUUGAAUUAGAGACGUUAAAAAUAAUGAAUCGACGUCGACUAUAACCC";
    std::string test4 = "GCUAGGGAUAACAGGGUGCGACCUGCCAAGCUGCACAAUUCAAUGUGGUUAGAAAACCAACUUGGAAUCCAAUCUCCAUGAGCCUACCAUCACAUGCGUUCUAGGGUUAACCUGAAGGUGUGAAGCUGAUGGGAAAAAGUAACCCAAACUGUAUGUGACAGUGAGGGGGCAGGCUAGAUUCCUAUGGGCAAUGUAAAUGAACACUCAUCUGAGGCAUCGUGACCCUAUCACAUCUAGUUAAUUGUGAGAGAAUCUUAUGUCUCUGUUUCAUAAGAUUGAUUGGACAAUUUCUCACCGGUGUAAAGAGUGGUCCUAAGGGAAUCAUCGAAAGUGAAUUGUGCGGAACAGGGCAAGCCCCAUAGGCUCCUUCGGGAGUGAGCGAAGCAAUUCUCUCUAUCGCCUAGUGGGUAAAAGACAGGGCAAAAAGCGAAUGCCAGUUAGUAAUAGACUGGAUAGGGUGAAUAACCUAACCUGAAAGGGUGCAGACUUGCUCAUGGGCGUGGGAAAUCAGAUUUCGUGAAUACACCAGCAUUCAAGAGUUUCCAUGCUUGAGCCGUGUGCGGUGAAAGUCGCAUGCACGGUUCUACUGGGGGGAAAGCCUGAGAGGGCCUACCUAUCCAACUUU";
    // std::string test5 = "AGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUA";


    // std::cout << "total(\"" << test1 << "\") = " << total(test1) << "\n";
    // std::cout << "total(\"" << test2 << "\") = " << total(test2) << "\n";
    std::cout << "total(\"" << test3 << "\") = " << total(test3) << "\n";
    std::cout << "total(\"" << test4 << "\") = " << total(test4) << "\n";

    return 0;
}
