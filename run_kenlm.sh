GRAM=2

tool/kenlm/build/bin/lmplz -o $GRAM --text data/wiki/wiki.seg.txt --arpa data/wiki/wiki.$GRAM.arpa

tool/kenlm/build/bin/build_binary data/wiki/wiki.$GRAM.arpa  data/wiki/wiki.$GRAM.bin
