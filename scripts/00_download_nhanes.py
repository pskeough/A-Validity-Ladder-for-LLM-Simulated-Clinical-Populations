"""Download the NHANES public microdata the analyses read, into data/nhanes_raw/, and check each file
against the SHA-256 of the copy the published results were computed from.

Cycles: 2005-2006 (D) to 2017-2018 (J), 2017-March 2020 pre-pandemic (P_), and August 2021-August 2023
(L). Files: demographics (DEMO), PHQ-9 (DPQ) and health insurance (HIQ). About 50 MB. Source: CDC,
https://wwwn.cdc.gov/nchs/nhanes/. NHANES public-use files are in the public domain.

A file whose hash differs is kept under its name with a warning: CDC occasionally reissues a file, and
a reissue can move published numbers slightly.
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "nhanes_raw")
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}"
YEAR = {"D": 2005, "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017, "L": 2021}

SHA256 = {
    "DEMO_D.xpt": "bf9cc79cce501b2fbe5aa8d5b403b618548d02d4aefeb528335cdd9f4c40a3af",
    "DEMO_E.xpt": "02b8e97703d04bc99769c751881bece51ac9ba68cf18a2992a45d3bdfb856147",
    "DEMO_F.xpt": "78c4d4a6e5eafa9da3bbb24a90d140fef5d8c0cda932a7ff8ee609288d4fae07",
    "DEMO_G.xpt": "eaf0525d1952626885af3e935415a1f66ad62c18698080e7354789c125af252d",
    "DEMO_H.xpt": "f8f0cbb3085a323d4cde22349b164878fea1e64dbc404e65b5815c7816b547d7",
    "DEMO_I.xpt": "c9297c6c37ae8f78f29be9568fa2a03cf3b112616a39afee04030fc775a66a0d",
    "DEMO_J.xpt": "c0b46e0345ea19404928656277c8b0d10b0cca348a9b2fe4fc3c67e8b7ee73ec",
    "DEMO_L.xpt": "ca4374a158b493b8b0163e1388da21d57a18d1b9cecff2aa4e2fa2bec494fe23",
    "DPQ_D.xpt": "d78e85a5ff46efc34d509897b85c1a87184f9153ed154ff01fe5085375a4c9e1",
    "DPQ_E.xpt": "05c63fc1e618b580ba2714011fb999161345faa3369eceb138e71d4dd7c49dea",
    "DPQ_F.xpt": "31fcc0a2df520b38e69ff42426201a6bb4755c59446cecc97b444663e62a3974",
    "DPQ_G.xpt": "38c3e2cb303c6f635330bcbbb0534df90e20ccedee23cf7622ca1ccbc6811844",
    "DPQ_H.xpt": "b07c49c34a1a91b986f5ca0a1b577eba08ac8ee307010be5a66fc1ef80d8788c",
    "DPQ_I.xpt": "6f4561aab077b9fd199a414fc516d6fedf40967266b81f1ab7b7d75afb72232d",
    "DPQ_J.xpt": "1c075f332e79736bd2360b41b0c10ed526883417e0b5996c407b1c564b01ca24",
    "DPQ_L.xpt": "25605a02685035fe997477b31a0991cca2776b0f72f24a8e61ac5b018456304b",
    "HIQ_D.xpt": "80b7d0399f765de6e970ac953740e478ae1379e2d4c95ec30c658c38951dc140",
    "HIQ_E.xpt": "e8bbdedfc943e578cb8e0d86aac2c9ab2a43e49bfe2613107d9a489749b949d5",
    "HIQ_F.xpt": "ab5a4bd89e003f31698e78fb5ae94e2a80a7bb407dcf014375d271466f04635b",
    "HIQ_G.xpt": "074351daf2979b949c72636f767fa4f9578ceaaee10fe6eb877462229f25d009",
    "HIQ_H.xpt": "1c2eec87f78f361cc8989973516bb8dd1b371da2cc70349dc6c84066c878a21e",
    "HIQ_I.xpt": "08ff064a49bb59f4ec78c6f97095a4fd42e1d55db7d8884fcb34f7c722e943c2",
    "HIQ_J.xpt": "96dfdd98507ea6dc67cb2c2b6999da3c035f3f8a9a87cf34756fe30762387860",
    "P_DEMO.xpt": "2e46c6c26bf77cd8989f64011ace12cbf42c0f3e03414eb59acc5328c8f87913",
    "P_DPQ.xpt": "b20ee66082eaea3d6d05c2699c142e9c49cf7737587364a477b82351cb8b9866",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    bad = 0
    for name, want in SHA256.items():
        path = os.path.join(OUT, name)
        if not os.path.exists(path):
            year = 2017 if name.startswith("P_") else YEAR[name[:-4].split("_")[-1]]
            req = urllib.request.Request(URL.format(year=year, name=name),
                                         headers={"User-Agent": "Mozilla/5.0 research"})
            with urllib.request.urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
        got = sha256(path)
        ok = got == want
        bad += not ok
        print(f"{'ok ' if ok else 'DIFFERS'}  {name}")
    if bad:
        print(f"{bad} file(s) differ from the copies the published results used", file=sys.stderr)


if __name__ == "__main__":
    main()
