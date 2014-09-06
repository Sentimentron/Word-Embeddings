Quick word-embeddings demonstration using [r9y9/nnet](https://github.com/r9y9/nnet).

[News article from WikiNews](https://en.wikinews.org/wiki/%27Misleading%27_Burger_King_advert_banned_in_the_United_Kingdom) is under a Creative Commons Attribution 2.5 License.

Does it work? Kind of. On the news article, the closest word representations (by Euclidean distance) are television, thickness, and complaints, which - given that the article's about people complaining about the size of a burger featured in a television advert - isn't totally wrong.

Todo list:
* Add support for custom loss functions in `mlp`
* Use this as an opportunity to profile and improve performance in `mlp`
* Tighten up the code for integration into [`golearn`](https://github.com/sjwhitworth/golearn).
* For some reason, I can't use the `LayerFuncs` defined in `nnet`, so that's something to fix.

License
-------

Copyright (c) 2014, Richard Townsend.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.