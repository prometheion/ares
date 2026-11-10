#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct node {
    int key;
    struct node *left, *right;
} node_t;

static node_t *insert(node_t *root, int key) {
    if (!root) {
        node_t *n = calloc(1, sizeof *n);
        if (n == NULL) { perror("calloc"); exit(EXIT_FAILURE); }
        n->key = key;
        return n;
    }
    if (key < root->key)      root->left  = insert(root->left, key);
    else if (key > root->key) root->right = insert(root->right, key);
    return root;
}

static void print_inorder(const node_t *root) {
    if (!root) return;
    print_inorder(root->left);
    printf("%d ", root->key);
    print_inorder(root->right);
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main(int argc, char **argv) {
    node_t *root = NULL;
    for (int i = 1; i < argc; ++i) root = insert(root, atoi(argv[i]));
    print_inorder(root);
    putchar('\n');
    unsigned long long big = 0xDEADBEEFull << 4 | 0x0F;
    printf("%llu %5.2f%%\n", big, 99.5);
    return MAX(0, 1) - 1;
}
