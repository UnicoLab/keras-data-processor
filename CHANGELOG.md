## 1.12.0 (2026-08-27)

* fix(KDP): correct distribution transforms, dynamic pipeline and distribution hints ([9f73c70](https://github.com/UnicoLab/keras-data-processor/commit/9f73c70))
* fix(KDP): make saved preprocessing models reloadable ([bf6bb16](https://github.com/UnicoLab/keras-data-processor/commit/bf6bb16))
* fix(KDP): make the docstring documentation generator run ([cd90054](https://github.com/UnicoLab/keras-data-processor/commit/cd90054))
* fix(KDP): make the standalone entry points work ([112ed4e](https://github.com/UnicoLab/keras-data-processor/commit/112ed4e))
* fix(KDP): only compute dataset statistics when a feature needs them ([10dcc5b](https://github.com/UnicoLab/keras-data-processor/commit/10dcc5b))
* fix(KDP): reject vocabulary statistics written by older releases ([7ecce75](https://github.com/UnicoLab/keras-data-processor/commit/7ecce75))
* fix(KDP): repair core defects, enable the linter and move to the Keras 3 API (#36) ([053c72d](https://github.com/UnicoLab/keras-data-processor/commit/053c72d)), closes [#36](https://github.com/UnicoLab/keras-data-processor/issues/36)
* fix(KDP): repair custom pipelines, inference conversion and the examples ([ac836dc](https://github.com/UnicoLab/keras-data-processor/commit/ac836dc))
* fix(KDP): repair four defects that only appeared when the model was called ([f0c1850](https://github.com/UnicoLab/keras-data-processor/commit/f0c1850)), closes [passthrou#excluded](https://github.com/passthrou/issues/excluded)
* fix(KDP): time series layers returned canned values instead of computing ([f584849](https://github.com/UnicoLab/keras-data-processor/commit/f584849))
* docs(KDP): add an upgrade guide for the 1.11.x transition ([9daadf3](https://github.com/UnicoLab/keras-data-processor/commit/9daadf3))
* docs(KDP): add Discord community link to README for user engagement ([4fc132d](https://github.com/UnicoLab/keras-data-processor/commit/4fc132d))
* docs(KDP): document the remaining public API and reach a clean lint gate ([ddfc62e](https://github.com/UnicoLab/keras-data-processor/commit/ddfc62e))
* docs(KDP): remove outdated badge links and streamline README content ([bfc4f5b](https://github.com/UnicoLab/keras-data-processor/commit/bfc4f5b))
* docs(KDP): update Discord community link in README for improved user access ([692fb24](https://github.com/UnicoLab/keras-data-processor/commit/692fb24))
* docs(KDP): update installation instructions in README to use 'kdp' instead of 'keras-data-processor' ([4966d86](https://github.com/UnicoLab/keras-data-processor/commit/4966d86))
* ci(KDP): fix release conditions, docs build and workflow dependencies ([f6ded58](https://github.com/UnicoLab/keras-data-processor/commit/f6ded58))
* ci(KDP): guard Keras 3 forward compatibility ([8adeeca](https://github.com/UnicoLab/keras-data-processor/commit/8adeeca))
* refactor(KDP): clear the lint backlog the disabled config had hidden ([22b601c](https://github.com/UnicoLab/keras-data-processor/commit/22b601c))
* refactor(KDP): target the Keras 3 API directly and fix what that exposed ([ed8d1e3](https://github.com/UnicoLab/keras-data-processor/commit/ed8d1e3))
* refactor(KDP): update repository links and branding to reflect UnicoLab ownership ([fcf82b9](https://github.com/UnicoLab/keras-data-processor/commit/fcf82b9))
* build(KDP): fix packaging metadata, docs links and destructive Makefile ([113498c](https://github.com/UnicoLab/keras-data-processor/commit/113498c))
* build(KDP): make the linter actually run ([6c38fb0](https://github.com/UnicoLab/keras-data-processor/commit/6c38fb0))
* chore: save last release version for recovery [skip ci] ([933d411](https://github.com/UnicoLab/keras-data-processor/commit/933d411))

## <small>1.11.2 (2025-07-30)</small>

* fix(KDP): adding docs for passthrough ([7164ce2](https://github.com/piotrlaczkowski/keras-data-processor/commit/7164ce2))
* fix(KDP): implementing reall passthrough (skipping all preocessing but adding inputs and ouputs ([3540a70](https://github.com/piotrlaczkowski/keras-data-processor/commit/3540a70))
* fix(KDP): preserve dtype for passthrough features and handle empty categorical vocabulary ([cc3d533](https://github.com/piotrlaczkowski/keras-data-processor/commit/cc3d533))
* chore: save last release version for recovery [skip ci] ([04a879a](https://github.com/piotrlaczkowski/keras-data-processor/commit/04a879a))

## <small>1.11.1 (2025-07-30)</small>

* fix(KDP): fixing tests ([6326dbf](https://github.com/piotrlaczkowski/keras-data-processor/commit/6326dbf))
* fix(KDP): formatting issues fixes ([6c60aed](https://github.com/piotrlaczkowski/keras-data-processor/commit/6c60aed))
* fix(KDP): increasing package version ([ce0dbf3](https://github.com/piotrlaczkowski/keras-data-processor/commit/ce0dbf3))
* fix(KDP): Preserve original dtype for PASSTHROUGH features (#30) ([82b6d7e](https://github.com/piotrlaczkowski/keras-data-processor/commit/82b6d7e)), closes [#30](https://github.com/piotrlaczkowski/keras-data-processor/issues/30)
* fix(KDP): update upload-artifact action to v4 in GitHub workflow ([68ee7c5](https://github.com/piotrlaczkowski/keras-data-processor/commit/68ee7c5))
* Add preserve dtype layer and update passthrough feature handling ([a700bad](https://github.com/piotrlaczkowski/keras-data-processor/commit/a700bad))
* Add pytest markers and improve test categorization for GitHub workflow ([06b5112](https://github.com/piotrlaczkowski/keras-data-processor/commit/06b5112))
* Checkpoint before follow-up message ([47ec0ef](https://github.com/piotrlaczkowski/keras-data-processor/commit/47ec0ef))
* chore: save last release version for recovery [skip ci] ([84e0b1f](https://github.com/piotrlaczkowski/keras-data-processor/commit/84e0b1f))
* refactor(KDP): improving tests execution ([b9d237e](https://github.com/piotrlaczkowski/keras-data-processor/commit/b9d237e))

## 1.11.0 (2025-05-28)

* feat(KDP): adding new layers / features to timeseries ([519bb06](https://github.com/piotrlaczkowski/keras-data-processor/commit/519bb06))
* feat(KDP): adding timeseries features ([7e819dc](https://github.com/piotrlaczkowski/keras-data-processor/commit/7e819dc))
* feat(KDP): adding timeseries features functionality and layers (#29) ([ff38437](https://github.com/piotrlaczkowski/keras-data-processor/commit/ff38437)), closes [#29](https://github.com/piotrlaczkowski/keras-data-processor/issues/29)
* feat(KDP): adding timeseries inference formatter ([c83cd48](https://github.com/piotrlaczkowski/keras-data-processor/commit/c83cd48))
* fix(KDP): fix pre-commit ([e1f77f4](https://github.com/piotrlaczkowski/keras-data-processor/commit/e1f77f4))
* fix(KDP): fixing docs plugin ([21dba63](https://github.com/piotrlaczkowski/keras-data-processor/commit/21dba63))
* fix(KDP): fixing symbolic tensors shape calculations ([d810e62](https://github.com/piotrlaczkowski/keras-data-processor/commit/d810e62))
* fix(KDP): getting rid of hardcoded test requirements ([849a982](https://github.com/piotrlaczkowski/keras-data-processor/commit/849a982))
* fix(KDP): removing print and replacing by logger ([2333152](https://github.com/piotrlaczkowski/keras-data-processor/commit/2333152))
* docs(KDP): correctly referencing timeseries features ([13e4511](https://github.com/piotrlaczkowski/keras-data-processor/commit/13e4511))
* docs(KDP): extending docs and tests for timeseries features ([37f4f24](https://github.com/piotrlaczkowski/keras-data-processor/commit/37f4f24))
* test(KDP): adding more timeseries feature tests ([8c317bf](https://github.com/piotrlaczkowski/keras-data-processor/commit/8c317bf))
* ops(KDP): cleaning workflows ([37da166](https://github.com/piotrlaczkowski/keras-data-processor/commit/37da166))
* ops(KDP): fix docs release errors ([b8972c4](https://github.com/piotrlaczkowski/keras-data-processor/commit/b8972c4))
* ops(KDP): fixing release ([37474e7](https://github.com/piotrlaczkowski/keras-data-processor/commit/37474e7))
* ops(KDP): improve release workflow ([2504321](https://github.com/piotrlaczkowski/keras-data-processor/commit/2504321))
* ops(KDP): updating after release notes ([cc41dad](https://github.com/piotrlaczkowski/keras-data-processor/commit/cc41dad))

## 1.10.0 (2025-03-31)

* ops(KDP): fixing action wrapper version ([cae2785](https://github.com/piotrlaczkowski/keras-data-processor/commit/cae2785))
* ops(KDP): fixing actions and node versions ([28cd93b](https://github.com/piotrlaczkowski/keras-data-processor/commit/28cd93b))
* ops(KDP): fixing semantic release version problem ([c6b612e](https://github.com/piotrlaczkowski/keras-data-processor/commit/c6b612e))
* ops(KDP): improving release workflows ([3f91e19](https://github.com/piotrlaczkowski/keras-data-processor/commit/3f91e19))
* ops(KDP): increating python version allowance ([1d06b76](https://github.com/piotrlaczkowski/keras-data-processor/commit/1d06b76))
* ops(KDP): increating python version allowance ([6070e69](https://github.com/piotrlaczkowski/keras-data-processor/commit/6070e69))
* ops(KDP): relaxing python requirements ([fa903e4](https://github.com/piotrlaczkowski/keras-data-processor/commit/fa903e4))
* docs(KDP): add an example_usages ([e980e12](https://github.com/piotrlaczkowski/keras-data-processor/commit/e980e12))
* docs(KDP): add to the docs to showcase new features ([790f102](https://github.com/piotrlaczkowski/keras-data-processor/commit/790f102))
* docs(KDP): added a diagram ([48d2e57](https://github.com/piotrlaczkowski/keras-data-processor/commit/48d2e57))
* docs(KDP): added docs ([d8da4c5](https://github.com/piotrlaczkowski/keras-data-processor/commit/d8da4c5))
* docs(KDP): added docs ([dd728ef](https://github.com/piotrlaczkowski/keras-data-processor/commit/dd728ef))
* docs(KDP): adding concrete examples of usage of advance features to docs (#21) ([ea419cf](https://github.com/piotrlaczkowski/keras-data-processor/commit/ea419cf)), closes [#21](https://github.com/piotrlaczkowski/keras-data-processor/issues/21)
* docs(KDP): adding new styling ([55ae7ce](https://github.com/piotrlaczkowski/keras-data-processor/commit/55ae7ce))
* docs(KDP): adding some images and API documentation tools ([a4be43f](https://github.com/piotrlaczkowski/keras-data-processor/commit/a4be43f))
* docs(KDP): adjusting numerical embeddings docs ([45f8872](https://github.com/piotrlaczkowski/keras-data-processor/commit/45f8872))
* docs(KDP): fixing missing links ([076c9b3](https://github.com/piotrlaczkowski/keras-data-processor/commit/076c9b3))
* docs(KDP): improving documentation ([02a137a](https://github.com/piotrlaczkowski/keras-data-processor/commit/02a137a))
* docs(KDP): improving documentation ([91d4f85](https://github.com/piotrlaczkowski/keras-data-processor/commit/91d4f85))
* docs(KDP): improving some remaining docs ([0c932b0](https://github.com/piotrlaczkowski/keras-data-processor/commit/0c932b0))
* docs(KDP): refactoring documentation ([1ccc3f2](https://github.com/piotrlaczkowski/keras-data-processor/commit/1ccc3f2))
* docs(KDP): removing whta we do not have ([93706fb](https://github.com/piotrlaczkowski/keras-data-processor/commit/93706fb))
* docs(KDP): reorganising documentation for better UX ([a67c634](https://github.com/piotrlaczkowski/keras-data-processor/commit/a67c634))
* docs(KDP): reorganising documentation for better UX ([052454e](https://github.com/piotrlaczkowski/keras-data-processor/commit/052454e))
* docs(KDP): revamping entire docs ([d0ef7b7](https://github.com/piotrlaczkowski/keras-data-processor/commit/d0ef7b7))
* docs(KDP): smart processing for custom pipelines ([8e7d0a7](https://github.com/piotrlaczkowski/keras-data-processor/commit/8e7d0a7))
* docs(KDP): updating DistributionEncoder docs ([1916c40](https://github.com/piotrlaczkowski/keras-data-processor/commit/1916c40))
* feat(KDP): add the AdvancedNumericalEmbedding feature ([8fa90e7](https://github.com/piotrlaczkowski/keras-data-processor/commit/8fa90e7))
* feat(KDP): added option to specify the distribution manually ([d3cce76](https://github.com/piotrlaczkowski/keras-data-processor/commit/d3cce76))
* feat(KDP): added selective retention of outputs based on dependencies among layers ([c17acd2](https://github.com/piotrlaczkowski/keras-data-processor/commit/c17acd2))
* feat(KDP): addin DistributionAwareEncored layer and numeric preprocessing ([9bfe276](https://github.com/piotrlaczkowski/keras-data-processor/commit/9bfe276))
* feat(KDP): adding auto config / recommender ([00e75d6](https://github.com/piotrlaczkowski/keras-data-processor/commit/00e75d6))
* feat(KDP): adding categorical features hashing ([078df9d](https://github.com/piotrlaczkowski/keras-data-processor/commit/078df9d))
* feat(KDP): adding DistributionAwareEncored layer for numeric features preprocessing. (#20) ([c988087](https://github.com/piotrlaczkowski/keras-data-processor/commit/c988087)), closes [#20](https://github.com/piotrlaczkowski/keras-data-processor/issues/20)
* feat(KDP): adding feature selection mechanism to the preprocessor (docs, tests) (#19) ([462bfc1](https://github.com/piotrlaczkowski/keras-data-processor/commit/462bfc1)), closes [#19](https://github.com/piotrlaczkowski/keras-data-processor/issues/19)
* feat(KDP): adding Gater Residual Variable / Features Selection Network capability ([f82b788](https://github.com/piotrlaczkowski/keras-data-processor/commit/f82b788))
* feat(KDP): adding MoE feature and tests ([fac7806](https://github.com/piotrlaczkowski/keras-data-processor/commit/fac7806))
* feat(KDP): adding numerical embedding layers (#26) ([5c3a974](https://github.com/piotrlaczkowski/keras-data-processor/commit/5c3a974)), closes [#26](https://github.com/piotrlaczkowski/keras-data-processor/issues/26)
* feat(KDP): adding passthrough feature ([2965916](https://github.com/piotrlaczkowski/keras-data-processor/commit/2965916))
* feat(kdp): adding TabularAttentionLayers and implementation ([f567585](https://github.com/piotrlaczkowski/keras-data-processor/commit/f567585))
* feat(kdp): adding TabularAttentionLayers and implementation (#11) ([cfbd38b](https://github.com/piotrlaczkowski/keras-data-processor/commit/cfbd38b)), closes [#11](https://github.com/piotrlaczkowski/keras-data-processor/issues/11)
* feat(KDP): Enhance Dynamic Preprocessing Pipeline (#24) ([bd90f11](https://github.com/piotrlaczkowski/keras-data-processor/commit/bd90f11)), closes [#24](https://github.com/piotrlaczkowski/keras-data-processor/issues/24)
* feat(KDP): global embedding for numeric features option added ([83f6996](https://github.com/piotrlaczkowski/keras-data-processor/commit/83f6996))
* feat(KDP): Integrate Advanced Numerical Embedding (#25) ([185292c](https://github.com/piotrlaczkowski/keras-data-processor/commit/185292c)), closes [#25](https://github.com/piotrlaczkowski/keras-data-processor/issues/25)
* feat(KDP): smart processing for custom pipelines ([448f63f](https://github.com/piotrlaczkowski/keras-data-processor/commit/448f63f))
* fix(KDP): add new examples for tabular attention cases and more complex Mixed Transformers and Tabul ([16340f2](https://github.com/piotrlaczkowski/keras-data-processor/commit/16340f2))
* fix(KDP): add transdormer() method to ProcessingModel ([0c6c65c](https://github.com/piotrlaczkowski/keras-data-processor/commit/0c6c65c))
* fix(KDP): added a missing code for the example for disttribution aware layer for custom pipelines ([9ca9fad](https://github.com/piotrlaczkowski/keras-data-processor/commit/9ca9fad))
* fix(KDP): added a missing code for the example for disttribution aware layer for custom pipelines ([9b92475](https://github.com/piotrlaczkowski/keras-data-processor/commit/9b92475))
* fix(KDP): added docstrings ([0eb968d](https://github.com/piotrlaczkowski/keras-data-processor/commit/0eb968d))
* fix(KDP): added fixes for the distribution estimator and tests ([da79bb9](https://github.com/piotrlaczkowski/keras-data-processor/commit/da79bb9))
* fix(KDP): Added get_feature_importances() method and fixed the docs. ([664023f](https://github.com/piotrlaczkowski/keras-data-processor/commit/664023f))
* fix(KDP): added prefered_distribution parameter for NumericalFeatures ([84b2eb5](https://github.com/piotrlaczkowski/keras-data-processor/commit/84b2eb5))
* fix(KDP): adding FeatureSelection to Text and Date features (#28) ([e1f453f](https://github.com/piotrlaczkowski/keras-data-processor/commit/e1f453f)), closes [#28](https://github.com/piotrlaczkowski/keras-data-processor/issues/28)
* fix(kdp): adding pre-commit fixes ([ec98d29](https://github.com/piotrlaczkowski/keras-data-processor/commit/ec98d29))
* fix(KDP): broke transform method into 2 separate methods and end-to-end tests ([11f258d](https://github.com/piotrlaczkowski/keras-data-processor/commit/11f258d))
* fix(KDP): changed the order of the transormers and the tabularAttention applications ([d387826](https://github.com/piotrlaczkowski/keras-data-processor/commit/d387826))
* fix(KDP): DistributionAwareEncoder fix and tests for custom pipelines (#23) ([ad91096](https://github.com/piotrlaczkowski/keras-data-processor/commit/ad91096)), closes [#23](https://github.com/piotrlaczkowski/keras-data-processor/issues/23)
* fix(KDP): edited some of the tests to reflect the changes in processor.py ([23b36ce](https://github.com/piotrlaczkowski/keras-data-processor/commit/23b36ce))
* fix(KDP): edited the docs ([34476c6](https://github.com/piotrlaczkowski/keras-data-processor/commit/34476c6))
* fix(KDP): Fix linter errors: unused variable and rearranged imports ([2c2a447](https://github.com/piotrlaczkowski/keras-data-processor/commit/2c2a447))
* fix(KDP): Fix remaining unused imports with ruff ([fe3a014](https://github.com/piotrlaczkowski/keras-data-processor/commit/fe3a014))
* fix(KDP): fix_tabukar_att_and_transfor_order_and_add_docs (#17) ([4b1c510](https://github.com/piotrlaczkowski/keras-data-processor/commit/4b1c510)), closes [#17](https://github.com/piotrlaczkowski/keras-data-processor/issues/17)
* fix(KDP): fixed all the algorithms for distribution detection all tests pass now ([52dad69](https://github.com/piotrlaczkowski/keras-data-processor/commit/52dad69))
* fix(KDP): Fixed dimensions micmatch for the input of the Tabular Attention ([5a66ad1](https://github.com/piotrlaczkowski/keras-data-processor/commit/5a66ad1))
* fix(KDP): fixed issues between graph and eager mode plus others ([a3fe7e1](https://github.com/piotrlaczkowski/keras-data-processor/commit/a3fe7e1))
* fix(KDP): fixes to the tests ([bc0c543](https://github.com/piotrlaczkowski/keras-data-processor/commit/bc0c543))
* fix(KDP): Fixing Distribution-Aware Encoder and adding comprehensive testing (#22) ([97b41c3](https://github.com/piotrlaczkowski/keras-data-processor/commit/97b41c3)), closes [#22](https://github.com/piotrlaczkowski/keras-data-processor/issues/22)
* fix(kdp): fixing docs requirements and release for docs ([9d8f8b3](https://github.com/piotrlaczkowski/keras-data-processor/commit/9d8f8b3))
* fix(KDP): fixing failiing tests ([966434d](https://github.com/piotrlaczkowski/keras-data-processor/commit/966434d))
* fix(kdp): fixing formatting issues ([352e72b](https://github.com/piotrlaczkowski/keras-data-processor/commit/352e72b))
* fix(KDP): fixing layers functionality ([ffe8d89](https://github.com/piotrlaczkowski/keras-data-processor/commit/ffe8d89))
* fix(kdp): fixing tests fromatting ([955ed08](https://github.com/piotrlaczkowski/keras-data-processor/commit/955ed08))
* fix(KDP): fixing the doc ([fdaa101](https://github.com/piotrlaczkowski/keras-data-processor/commit/fdaa101))
* fix(KDP): improving docs UX ([f51daf8](https://github.com/piotrlaczkowski/keras-data-processor/commit/f51daf8))
* fix(KDP): reformatting with pre-commits ([2f01e67](https://github.com/piotrlaczkowski/keras-data-processor/commit/2f01e67))
* fix(KDP): reformatting with pre-commits ([2523f89](https://github.com/piotrlaczkowski/keras-data-processor/commit/2523f89))
* fix(KDP): removed an unused method ([a170db8](https://github.com/piotrlaczkowski/keras-data-processor/commit/a170db8))
* fix(KDP): Removed some buggy feature ([da24b7b](https://github.com/piotrlaczkowski/keras-data-processor/commit/da24b7b))
* fix(KDP): Removing data.csv ([3ca0daf](https://github.com/piotrlaczkowski/keras-data-processor/commit/3ca0daf))
* fix(KDP): small fixes ([47a6267](https://github.com/piotrlaczkowski/keras-data-processor/commit/47a6267))
* test(KDP): add end to end and unit test for the "TabularAttention" and the "MultiResolutionTabularAt ([1cf5d09](https://github.com/piotrlaczkowski/keras-data-processor/commit/1cf5d09))
* test(kdp): add end to end tests ([a1b3018](https://github.com/piotrlaczkowski/keras-data-processor/commit/a1b3018))
* test(kdp): add end to end tests (#13) ([5737d9b](https://github.com/piotrlaczkowski/keras-data-processor/commit/5737d9b)), closes [#13](https://github.com/piotrlaczkowski/keras-data-processor/issues/13)
* test(KDP): add more tests ([a4d536c](https://github.com/piotrlaczkowski/keras-data-processor/commit/a4d536c))
* test(KDP): add more tests (#15) ([84bfdc6](https://github.com/piotrlaczkowski/keras-data-processor/commit/84bfdc6)), closes [#15](https://github.com/piotrlaczkowski/keras-data-processor/issues/15)
* test(KDP): add tests ([5cb4e8e](https://github.com/piotrlaczkowski/keras-data-processor/commit/5cb4e8e))
* test(KDP): add tests ([55cbbb3](https://github.com/piotrlaczkowski/keras-data-processor/commit/55cbbb3))
* test(KDP): add tests and fix (#14) ([5308029](https://github.com/piotrlaczkowski/keras-data-processor/commit/5308029)), closes [#14](https://github.com/piotrlaczkowski/keras-data-processor/issues/14)
* test(KDP): add tests for various cases and also a ValueError for missing vocab scenario ([82a99a3](https://github.com/piotrlaczkowski/keras-data-processor/commit/82a99a3))
* test(KDP): add unit test for gates res network ([1d980e0](https://github.com/piotrlaczkowski/keras-data-processor/commit/1d980e0))
* test(KDP): added test for advanced features ([d4fc5f3](https://github.com/piotrlaczkowski/keras-data-processor/commit/d4fc5f3))
* test(KDP): added test for advanced features ([c18a59b](https://github.com/piotrlaczkowski/keras-data-processor/commit/c18a59b))
* test(KDP): adding passthrough tests ([f98dc2c](https://github.com/piotrlaczkowski/keras-data-processor/commit/f98dc2c))
* test(KDP): dummy ([4181bb3](https://github.com/piotrlaczkowski/keras-data-processor/commit/4181bb3))
* test(KDP): dummy commit ([2883d79](https://github.com/piotrlaczkowski/keras-data-processor/commit/2883d79))
* test(KDP): dummy commit ([1f1e35b](https://github.com/piotrlaczkowski/keras-data-processor/commit/1f1e35b))
* test(KDP): empty commit for testing ([9b0f386](https://github.com/piotrlaczkowski/keras-data-processor/commit/9b0f386))
* test(KDP): extending testes for preprocessor module ([2e23b3d](https://github.com/piotrlaczkowski/keras-data-processor/commit/2e23b3d))
* refactor(KDP): impreoving auto configuration functionality and UX ([7b76a99](https://github.com/piotrlaczkowski/keras-data-processor/commit/7b76a99))
* refactor(KDP): maintainance on preprocessor to optimize code and refactor ([7146afe](https://github.com/piotrlaczkowski/keras-data-processor/commit/7146afe))
* refactor(KDP): removing tf-proba dependency ([83cb73d](https://github.com/piotrlaczkowski/keras-data-processor/commit/83cb73d))
* refactor(KDP): splitting custom_layers ([f029f77](https://github.com/piotrlaczkowski/keras-data-processor/commit/f029f77))
* refactor(KDP): splitting layers into separate files ([c188267](https://github.com/piotrlaczkowski/keras-data-processor/commit/c188267))
* refactor(KDP): splitting more tests for layers ([84293f0](https://github.com/piotrlaczkowski/keras-data-processor/commit/84293f0))
* feat(validation): add day of the month add assertions and error handling ([fa88c24](https://github.com/piotrlaczkowski/keras-data-processor/commit/fa88c24))
* fix: update distribution aware encoder and tests ([62f0dba](https://github.com/piotrlaczkowski/keras-data-processor/commit/62f0dba))
* fix(validation): added unit tests and fixed some small bugs (#12) ([3042e2a](https://github.com/piotrlaczkowski/keras-data-processor/commit/3042e2a)), closes [#12](https://github.com/piotrlaczkowski/keras-data-processor/issues/12)
* Merge branch 'main' into feat_adding_grvs ([147bceb](https://github.com/piotrlaczkowski/keras-data-processor/commit/147bceb))
* Merge branch 'main' into feat_dist_aware_embedding_numerical ([27dff94](https://github.com/piotrlaczkowski/keras-data-processor/commit/27dff94))
* Merge branch 'main' into fix_tab_att_and_transfor ([6312d09](https://github.com/piotrlaczkowski/keras-data-processor/commit/6312d09))
* Merge branch 'main' into tabular_attention_tests ([9491590](https://github.com/piotrlaczkowski/keras-data-processor/commit/9491590))
* Merge branch 'piotrlaczkowski:main' into cutom_preprocess_smart ([1d7e48a](https://github.com/piotrlaczkowski/keras-data-processor/commit/1d7e48a))
* Merge branch 'piotrlaczkowski:main' into feat_num_emb ([0bf0cc3](https://github.com/piotrlaczkowski/keras-data-processor/commit/0bf0cc3))
* Merge branch 'piotrlaczkowski:main' into feat_num_emb ([1dd8e83](https://github.com/piotrlaczkowski/keras-data-processor/commit/1dd8e83))
* Merge pull request #1 from piotrlaczkowski/main ([609dd5b](https://github.com/piotrlaczkowski/keras-data-processor/commit/609dd5b)), closes [#1](https://github.com/piotrlaczkowski/keras-data-processor/issues/1)
* test(validation): added unit tests and fixed a little type mismatch ([d0303cb](https://github.com/piotrlaczkowski/keras-data-processor/commit/d0303cb))

## [1.9.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.8.0...kdp.1.9.0) (2024-12-06)


### :gift: Features

* feat(kdp): migrating stats to pools executor ([ad2496a](https://github.com/piotrlaczkowski/keras-data-processor/commit/ad2496a05fba380f3327e9a1948d85d5a4fa4303))
* feat(kdp): adding parallelization + fixes ([788e7f5](https://github.com/piotrlaczkowski/keras-data-processor/commit/788e7f5248ba17e1ac000b86901ae3ad6a8529ba))


### :beetle: Bug Fixes

* fix(kdp): fixing tests + new coverage ([c9f895f](https://github.com/piotrlaczkowski/keras-data-processor/commit/c9f895ffca2e1be0668a1fdc56d25f65eaed4879))
* fix(kdp): reparing shapes and transformer blocks ([2334171](https://github.com/piotrlaczkowski/keras-data-processor/commit/23341712ad47cf70767c2f089b6c97c7e4b7074d))

## [1.8.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.7.0...kdp.1.8.0) (2024-08-19)


### :beetle: Bug Fixes

* fix(KDP): fixing functionality for missing categorical features ([62d10f9](https://github.com/piotrlaczkowski/keras-data-processor/commit/62d10f99d4d04c80f4e0de11b58d5aa22e2171c3))
* fix(KDP): fixing functionality for missing categorical features ([40c85d7](https://github.com/piotrlaczkowski/keras-data-processor/commit/40c85d7f1c085a0e7b1825c9822db8dd57828552))
* fix(KDP): fixing functionality for missing categorical features ([22593e9](https://github.com/piotrlaczkowski/keras-data-processor/commit/22593e9cd62df1bcd7fd96e16e2533bbc113ea68))


### :gift: Features

* feat(kdp): adding date preprocessing layers and core code (#7) ([3560933](https://github.com/piotrlaczkowski/keras-data-processor/commit/3560933b81d83e363877bc4452555895eba70207)), closes [#7](https://github.com/piotrlaczkowski/keras-data-processor/issues/7)
* feat(kdp): adding date preprocessing layers ([5ffa865](https://github.com/piotrlaczkowski/keras-data-processor/commit/5ffa8658f45fdd9d2f58203df59c4940e750599a))
* feat(kdp): adding date preprocessing layers ([f97e497](https://github.com/piotrlaczkowski/keras-data-processor/commit/f97e497deb0a4a31f28caff726aef04f3c5b77f2))
* feat(kdp): adding date preprocessing layers ([f4ae5b2](https://github.com/piotrlaczkowski/keras-data-processor/commit/f4ae5b27d5406cc6d52b6f65d0b2fd37d7626e28))
* feat(kdp): adding date preprocessing layers ([552f26b](https://github.com/piotrlaczkowski/keras-data-processor/commit/552f26b6d11ea082d2735ed094e164973f145bd0))
* feat(kdp): adding date preprocessing layers ([23eca47](https://github.com/piotrlaczkowski/keras-data-processor/commit/23eca4770be133c97d845d2f3a2816c9b94b3f2c))

## [1.7.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.6.0...kdp.1.7.0) (2024-07-08)


### :hotsprings: Infra

* ops(KDP): fixing u-tests action ([d4bdf24](https://github.com/piotrlaczkowski/keras-data-processor/commit/d4bdf24a17b1b6812835a0fb750ed6c3d1552633))
* ops(KDP): adding manual PyPI release action ([a192d08](https://github.com/piotrlaczkowski/keras-data-processor/commit/a192d08c3eb3adb4b9599e027bec5dfb6051427b))
* ops(KDP): updating to lates version in the config ([c0454ad](https://github.com/piotrlaczkowski/keras-data-processor/commit/c0454ad4ba02cb0987263db059635b68c4b81816))


### :tools: Maintenance

* refactor(KDP): simplifying layers factory creator (#5) ([b63a504](https://github.com/piotrlaczkowski/keras-data-processor/commit/b63a50419f01372196bd1be42801ec275ef75bf6)), closes [#5](https://github.com/piotrlaczkowski/keras-data-processor/issues/5)
* refactor(KDP): simplifying layers factory creator ([ee0c466](https://github.com/piotrlaczkowski/keras-data-processor/commit/ee0c466ee9bdaef533891ff9a0e763f3927c032c))


### :beetle: Bug Fixes

* fix(KDP): fixing functionality for missing categorical features ([bb569bf](https://github.com/piotrlaczkowski/keras-data-processor/commit/bb569bff62a9c22a7e44a91f02b503773246e484))

## [1.6.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.5.1...kdp.1.6.0) (2024-04-30)


### :gift: Features

* feat(KDP): adding TransformerBlocks to all Features Options ([e27b0e5](https://github.com/piotrlaczkowski/keras-data-processor/commit/e27b0e5880e4bb69deb50ad8d2a3ca0cb8c33eb7))
* feat(KDP): adding TransformerBlocks layers to categorical and text variables ([0338fb4](https://github.com/piotrlaczkowski/keras-data-processor/commit/0338fb4085312b6e63752821cd628f402dcd5607))


### :hotsprings: Infra

* ops(KDP): updating to lates version in the config ([5cf8998](https://github.com/piotrlaczkowski/keras-data-processor/commit/5cf8998374d4e7e7b2b7b33cefe6132597669650))

## [1.5.1](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.5.0...kdp.1.5.1) (2024-04-19)


### :beetle: Bug Fixes

* fix(KDP): adding missing text features vocab attribution ([d9cbc13](https://github.com/piotrlaczkowski/keras-data-processor/commit/d9cbc136f0dc282567dea9807d8993953269f44f))

## [1.5.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.4.0...kdp.1.5.0) (2024-04-18)


### :hotsprings: Infra

* ops(KDP): adjusting tests display on MR ([0459a68](https://github.com/piotrlaczkowski/keras-data-processor/commit/0459a68a64e7cd1ba82faa119d9b0292b12fa6e7))
* ops(KDP): disabeling prettier ([9cfab3a](https://github.com/piotrlaczkowski/keras-data-processor/commit/9cfab3a2a4b9a9a8089cf15e1a979bac572edc0d))


### :beetle: Bug Fixes

* fix(KDP): fixing output dim display for DICT output ([2c2b147](https://github.com/piotrlaczkowski/keras-data-processor/commit/2c2b147b3d904f878b0d6d2979be78559eb093c7))

## [1.4.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.3.0...kdp.1.4.0) (2024-04-15)


### :hotsprings: Infra

* ops(KDP): fixing version to 1.3 ([99eeabc](https://github.com/piotrlaczkowski/keras-data-processor/commit/99eeabc1cff425561f0205f15242d1ee458279a2))
* ops(KDP): fixing version to 1.3 ([3bd5b2f](https://github.com/piotrlaczkowski/keras-data-processor/commit/3bd5b2f3980de7e06bdf010788fd706fbb7b8111))

## [1.3.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.2.0...kdp.1.3.0) (2024-04-15)


### :beetle: Bug Fixes

* fix(KDP): fixing relative imports problem ([9ca89ac](https://github.com/piotrlaczkowski/keras-data-processor/commit/9ca89aca02685f1fd8a105243c7ee8fc02dc5475))
* fix(KDP): preprocess setting up base test + 1 ([8ef8ffa](https://github.com/piotrlaczkowski/keras-data-processor/commit/8ef8ffa7074b4b5d8a318e3df81349d1fef81854))
* fix(KDP): preprocess setting up base test ([9ee3fa6](https://github.com/piotrlaczkowski/keras-data-processor/commit/9ee3fa6f577ffe368e7f738d8a1566862e2ed304))
* fix(KDP): stats tests (adding todo) ([8883731](https://github.com/piotrlaczkowski/keras-data-processor/commit/8883731e841dd64ace5007ff7cc391fd62cd2f24))
* fix(KDP): pipeline tests (with TODO) ([a89accb](https://github.com/piotrlaczkowski/keras-data-processor/commit/a89accb31e2eee35d42ca1e64a9c00c64786d058))
* fix(KDP): fixing imports ([fbcb9d1](https://github.com/piotrlaczkowski/keras-data-processor/commit/fbcb9d104c25a8772d1af905cec0e456ebea4dd0))


### :tools: Maintenance

* refactor(KDP): adding cast_to_float32 layers to int outputs ([cc274df](https://github.com/piotrlaczkowski/keras-data-processor/commit/cc274df271cd2f5a6747d317d204d9bac5c30ba8))
* refactor(KDP): fixing imports ([b420ef0](https://github.com/piotrlaczkowski/keras-data-processor/commit/b420ef006505e8e3b99857f0e5a804a7d5a8269c))
* refactor(KDP): separating feature space normalizing class ([8b9c394](https://github.com/piotrlaczkowski/keras-data-processor/commit/8b9c39484aa2fafbdff182110e9f62bed7c70bf6))
* refactor(KDP): adding abstraction for custom pipeline steps ([d500474](https://github.com/piotrlaczkowski/keras-data-processor/commit/d500474b2c31a92855e1097acab217bcec3882c4))


### :gift: Features

* feat(KDP): adding custom pipelines ([90d098c](https://github.com/piotrlaczkowski/keras-data-processor/commit/90d098c8baa36fbd1a6c1e12f9e7b345f62d58d0))
* feat(KDP): adding float32 cast layer ([f2b02fb](https://github.com/piotrlaczkowski/keras-data-processor/commit/f2b02fb6cbf40f00003f83be3d6376ee27886955))
* feat(KDP): migrating to feature specs approach v1 ([8953da5](https://github.com/piotrlaczkowski/keras-data-processor/commit/8953da5241f66b738e9c7b51694f093b4ad3b725))
* feat(KDP): adding text processing option to the preprocessor v0 ([9e8dc03](https://github.com/piotrlaczkowski/keras-data-processor/commit/9e8dc03f13f12708958a9b6f720bb6e4b8682668))
* feat(KDP): adding text processing option to the preprocessor v0 ([c0724c4](https://github.com/piotrlaczkowski/keras-data-processor/commit/c0724c4180e37ae9a1b639017340eb930d54dccd))
* feat(KDP): adding text processing layers ([0a1e06d](https://github.com/piotrlaczkowski/keras-data-processor/commit/0a1e06d24c17aecaa6f5e5dc819d5b158d145452))

## [1.2.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.1.0...kdp.1.2.0) (2024-03-14)


### :hotsprings: Infra

* ops(KDP): adding docs link ([ee639af](https://github.com/piotrlaczkowski/keras-data-processor/commit/ee639afcd7fe4a69422e352f2499d649a74de9a8))
* ops(KDP): adding docs link ([447a680](https://github.com/piotrlaczkowski/keras-data-processor/commit/447a680b223d58d66fe7951b4b6e1a4ef12b8a92))

## [1.1.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.0.0...kdp.1.1.0) (2024-03-14)


### :hotsprings: Infra

* ops(KDP): adding docs link ([53d51ba](https://github.com/piotrlaczkowski/keras-data-processor/commit/53d51bab7145d7b5ccf899f06aecc59e87f24ffd))

## 1.0.0 (2024-03-14)


### :gift: Features

* feat(KDP): adding options enums + comments ([598e2e7](https://github.com/piotrlaczkowski/keras-data-processor/commit/598e2e7fe558684057bde9b54effc95294f78b7e))
* feat(KDP): adding options enums ([0ef86a7](https://github.com/piotrlaczkowski/keras-data-processor/commit/0ef86a7915b96186bb3c12c3143f29dc60cf83c8))
* feat(KDP): connecting custom embedding size dict to the code ([7e7e927](https://github.com/piotrlaczkowski/keras-data-processor/commit/7e7e927839428e05b40ff9affda53eb8b5f1912d))
* feat(KDP): adding concat layer output formation ([0fe667f](https://github.com/piotrlaczkowski/keras-data-processor/commit/0fe667f92b52df35b7f31d1891457c1b45f226b5))
* feat(KDP): adding string decoding and sorting ([c057516](https://github.com/piotrlaczkowski/keras-data-processor/commit/c057516bb46081f9dc6efb208bb72448b5b4ed4e))
* feat(KDP): implementing accumulators based on  Welford method ([e91495c](https://github.com/piotrlaczkowski/keras-data-processor/commit/e91495cee7658909726e0cdbf3eb8fac219ed1c9))
* feat(KDP): implementing accumulators based on  Welford method ([1bfb31d](https://github.com/piotrlaczkowski/keras-data-processor/commit/1bfb31db780bd2553f85d805c40ba236e817c02e))
* init(KDP): base code setup ([a084fab](https://github.com/piotrlaczkowski/keras-data-processor/commit/a084fab7712b7f908df649bcb4082cba2654ff6b))


### :tools: Maintenance

* refactor(KDP): splitting into multiple files ([8472676](https://github.com/piotrlaczkowski/keras-data-processor/commit/8472676513cd17cf6cce98e3def976e4d29b1e91))
* refactor(KDP): splitting into multiple files ([10772e9](https://github.com/piotrlaczkowski/keras-data-processor/commit/10772e96947a97d3bfdaff8581b6bdf1a4847467))
* refactor(KDP): adding crosses and bucketized cols ([570553d](https://github.com/piotrlaczkowski/keras-data-processor/commit/570553dcf6352d90200493f2e60bf004f4dd0a84))
* refactor(KDP): adding first functional pipeline connections ([2b1fa04](https://github.com/piotrlaczkowski/keras-data-processor/commit/2b1fa048848bc47df3069c844e25d221f692185d))
* refactor(KDP): adding new structure ([bb81977](https://github.com/piotrlaczkowski/keras-data-processor/commit/bb81977a184af4c4bf2262d655f6a739f99dadf7))


### :hotsprings: Infra

* ops(KDP): reformatting settings ([9df78f3](https://github.com/piotrlaczkowski/keras-data-processor/commit/9df78f3847aa720a499028b491e2641c33fdee23))
* ops(KDP): reformatting settings ([a148903](https://github.com/piotrlaczkowski/keras-data-processor/commit/a148903536b3735efcd1fcc919417ea00be789fd))
* ops(KDP): reformatting settings ([1cd4f3c](https://github.com/piotrlaczkowski/keras-data-processor/commit/1cd4f3c8f52c3f055718a7c68752da9ae732668d))
* ops(KDP): attaching correct workflos container and correcting owner ([8ae1f2d](https://github.com/piotrlaczkowski/keras-data-processor/commit/8ae1f2da8062d60b8d558e8a0df0679faa419ce0))
* ops(KDP): attaching correct workflos container ([f64f0fa](https://github.com/piotrlaczkowski/keras-data-processor/commit/f64f0faba1696f4085ed6ee581de0e4fcdc6f636))
* ops(KDP): adding direct imports ([de6f215](https://github.com/piotrlaczkowski/keras-data-processor/commit/de6f215833d7f1d685b0a7df96572b5c141c20d8))
* ops(KDP): adding Makefile ([599c75a](https://github.com/piotrlaczkowski/keras-data-processor/commit/599c75a5917ce49434b9cbb6f6469a1a184b001e))
* ops(KDP): adding Makefile ([4f9cf2a](https://github.com/piotrlaczkowski/keras-data-processor/commit/4f9cf2af186731bf3bdb2fc92b5c0d520b082447))
* ops(KDP): adding more exception to gitignore ([4b4ce7d](https://github.com/piotrlaczkowski/keras-data-processor/commit/4b4ce7da47407bd57a613c5946f533b66404dd73))
* ops(KDP): adding pydot for model plotting (group dev) ([8be28a7](https://github.com/piotrlaczkowski/keras-data-processor/commit/8be28a772f64a822e4cd8d23562a89b7e36bea1d))
* ops(KDP): adding pandas to dev dependencies ([d151adc](https://github.com/piotrlaczkowski/keras-data-processor/commit/d151adcb59b812dc7a32020a428c2046dc9e7b96))
* ops(KDP): adjusting workflows ([a544e72](https://github.com/piotrlaczkowski/keras-data-processor/commit/a544e720d0a7f672179af04214fce948f5e3de80))
