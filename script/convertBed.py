from path import Path

filename = Path("test.hits-vs-ref.hits.log2-0.6.pvalue-0.001.minw-4.cnv")
filename2 = Path("cnvseq.bed")
print("start")
with open(filename) as f, open(filename2, "w") as w:
    _ = None
    for line in f:
        if not _:
            _ = 54
            continue
        items = line.split()
        if items[6] == "NA":
            continue
        final = " ".join([items[0], items[5], str(int(items[5])+1), items[6]])
        final = "".join([final, "\n"])
        w.write(final)

print("end")
