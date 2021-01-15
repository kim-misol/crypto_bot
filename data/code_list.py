coin_codes = ['BTC/KRW', 'ETH/KRW', 'XRP/KRW', 'ADA/KRW']

stock_codes = ['155660', '001250', '294870', '001390', '011070', '066570', '037560', '010060', '100840', '064960', '096770',
         '011810', '024070', '037270', '011420', '010130', '014530', '008350', '353200', '014160', '003090', '006650',
         '000150', '241560', '115390', '286940', '027740', '204320', '008560', '204210', '001270', '006400', '028050',
         '010140', '068290', '000810', '000070', '005680', '003720', '001290', '007610', '136490', '004490', '091090',
         '293940', '055550', '003620', '011090', '012280', '008600', '033270', '023800', '129260', '013360', '081000',
         '271980', '001560', '002620', '002600', '071320', '088790', '120110', '020120', '015890', '004100', '058430',
         '033180', '005430', '023350', '168490', '009180', '002220', '006390', '009460', '088350', '267270', '012330',
         '004560', '017800', '001500', '057050', '111110', '079160', '097950', '089470', '096760', '105560', '119650',
         '001380', '014280', '002990', '001210', '024110', '004270', '251270', '006370', '047040', '003220', '001070',
         '001790', '005880', '024900', '018500', '042670', '024090', '007340', '000400', '280360', '005300', '011170',
         '071840', '107590', '035150', '002760', '003960', '008040', '003230', '021050', '004360', '058650', '067830',
         '006880', '002700', '019170', '004770', '122900', '003560', '078520', '036570', '014440', '003520', '000670',
         '047400', '000220', '000100', '139480', '005950', '015020', '034590', '008500', '001630', '013870', '011000',
         '109070', '281820', '007810', '005420', '071950', '138490', '192400', '015590', '084870', '003670', '019490',
         '071050', '006200', '161390', '034830', '007280', '009830', '101530', '322000', '011210', '227840', '008770',
         '006060', '000850', '016580', '004800', '298000', '003280', '282330', '138930', '004840', '006360', '039570',
         '025000', '058860', '001120', '023150', '034120', '101060', '017670', '035250', '030610', '002720', '001570',
         '002350', '005250', '072710', '058730', '019680', '001880', '000210', '069620', '012510', '004830', '004140',
         '007590', '026960', '102260', '000640', '001520', '082640', '006040', '003160', '013570', '026890', '004000',
         '002270', '088980', '090370', '017180', '009680', '357250', '134380', '002410', '096300', '005180', '207940',
         '018260', '016360', '029780', '002170', '002810', '009770', '248170', '200880', '003080', '306200', '308170',
         '004430', '011930', '031430', '031440', '001770', '004920', '183190', '018250', '244920', '181710', '011330',
         '003460', '025820', '084680', '003120', '194370', '025620', '001550', '120030', '063160', '013890', '051630',
         '145270', '357120', '005070', '021240', '284740', '012170', '007980', '055490', '078000', '363280', '036580',
         '016800', '005810', '071090', '000080', '004090', '002200', '123890', '025890', '024720', '000240', '047810',
         '060980', '008930', '009240', '004150', '011700', '000880', '012450', '005380', '126560', '094280', '005870',
         '000120', '069730', '365550', '114090', '001060', '092230', '044450', '093050', '108670', '051910', '079550',
         '006260', '035420', '034310', '008260', '005090', '002360', '001510', '000660', '012610', '009140', '001140',
         '092440', '006280', '008060', '008110', '005750', '009190', '012800', '001230', '002900', '000020', '034020',
         '001530', '032350', '094800', '033920', '007120', '155900', '005030', '030790', '014710', '028260', '005930',
         '023000', '004380', '000520', '005500', '004450', '007860', '002820', '004980', '002420', '019440', '013000',
         '021820', '075580', '145210', '068270', '004970', '005390', '004170', '090430', '002030', '010780', '123700',
         '140910', '138250', '085310', '004250', '111770', '009970', '118000', '017370', '105840', '049800', '014830',
         '001200', '008250', '214320', '102460', '006490', '226320', '004910', '000480', '272450', '009310', '344820',
         '003070', '006890', '214420', '001020', '005490', '007630', '293480', '002960', '104700', '003350', '053690',
         '010420', '001750', '097230', '003480', '267260', '004020', '032560', '095570', '006840', '001040', '011150',
         '016610', '139130', '017940', '007700', '092220', '001940', '003550', '034220', '229640', '338100', '003570',
         '036530', '010950', '001740', '285130', '012320', '013580', '005320', '003920', '025860', '090350', '001680',
         '016710', '084010', '069460', '145720', '049770', '336260', '092200', '001080', '009900', '005360', '008420',
         '085620', '003650', '001340', '007210', '026940', '015350', '002070', '100220', '007160', '001470', '006660',
         '032830', '009150', '001360', '145990', '272550', '004440', '009470', '007540', '008490', '001430', '027970',
         '017550', '034300', '005800', '001720', '009270', '002870', '033660', '005850', '007460', '900140', '010120',
         '271560', '316140', '006980', '010600', '000910', '003470', '103590', '004700', '018470', '002780', '035720',
         '009070', '030200', '003690', '044820', '144620', '264900', '005740', '039490', '011280', '001420', '019180',
         '028670', '090080', '039130', '172580', '010100', '002390', '069640', '213500', '014680', '004710', '018880',
         '002320', '195870', '079430', '069960', '093240', '003010', '002460', '079980', '027410', '000590', '012030',
         '007070', '012630', '234080', '009440', '000040', '032640', '005610', '002100', '267290', '000050', '002240',
         '009290', '037710', '007690', '008870', '073240', '006570', '003540', '000430', '024890', '002880', '192080',
         '005960', '002210', '170900', '084670', '008970', '030720', '023530', '000060', '268280', '003000', '011390',
         '002450', '004690', '004410', '000180', '033530', '012600', '134790', '267850', '020560', '001780', '025530',
         '007310', '002920', '011690', '088260', '007660', '093230', '014990', '000230', '015860', '317400', '033240',
         '000950', '348950', '036420', '030000', '035000', '010640', '002380', '192820', '031820', '014580', '091810',
         '010820', '047050', '153360', '152550', '010040', '000970', '123690', '014790', '042700', '020000', '105630',
         '025750', '009420', '130660', '000370', '143210', '307950', '241590', '133820', '010660', '298050', '093370',
         '081660', '005010', '000990', '082740', '016380', '030210', '058850', '051900', '004060', '009160', '011790',
         '018670', '210980', '068400', '077970', '071970', '002710', '000500', '214390', '012200', '002140', '017900',
         '000270', '013700', '005720', '003580', '004370', '023590', '128820', '000300', '001440', '001130', '003490',
         '003830', '023450', '282690', '092780', '014820', '210540', '330590', '138040', '009580', '002840', '003610',
         '003850', '006090', '006110', '010960', '011230', '000390', '075180', '014910', '003030', '336370', '035510',
         '102280', '003410', '008700', '002310', '161000', '023960', '298690', '015260', '009810', '012160', '002630',
         '001800', '070960', '016880', '005820', '077500', '074610', '334890', '101140', '249420', '007110', '007570',
         '044380', '010580', '003780', '001620', '029460', '053210', '002020', '004870', '005690', '010770', '017810',
         '000140', '025540', '009540', '017960', '011500', '128940', '004960', '014130', '052690', '272210', '003530',
         '086280', '267250', '001450', '013520', '298040', '298020', '001460', '005830', '078930', '011200', '175330',
         '000680', '005940', '034730', '006120', '000860', '009450', '017040', '339770', '083420', '011780', '214330',
         '004540', '001260', '000320', '000490', '084690', '117580', '009320', '042660', '006340', '015230', '016090',
         '002150', '005190', '028100', '163560', '004890', '002690', '016740', '192650', '004990', '012690', '009200',
         '025560', '006800', '000890', '352820', '001820', '041650', '017390', '011300', '016590', '029530', '004080',
         '015540', '112610', '002790', '012750', '326030', '003060', '006740', '015360', '004720', '010050', '010400',
         '095720', '000700', '072130', '008730', '350520', '000760', '023810', '003200', '020760', '020150', '006220',
         '089590', '185750', '100250', '000650', '033250', '006380', '033780', '003240', '009410', '103140', '086790',
         '036460', '015760', '161890', '003680', '016450', '300720', '003300', '051600', '180640', '005110', '000720',
         '005440', '064350', '010620', '004310', '011760', '010690', '069260', '000540', '245620', '083450', '028150',
         '351340', '099520', '067290', '024120', '060720', '337450', '340350', '051390', '024910', '198440', '215000',
         '307280', '043650', '204020', '286750', '085910', '007390', '031390', '012340', '085670', '078140', '369370',
         '290380', '290120', '003310', '023910', '299170', '006620', '099410', '196490', '066900', '127120', '092070',
         '033130', '060240', '219420', '086900', '021880', '327260', '080160', '250060', '118990', '028040', '207760',
         '214610', '206640', '053030', '337930', '023600', '111870', '009300', '042940', '100660', '080470', '011560',
         '258830', '053450', '049180', '208370', '068760', '053110', '204630', '013810', '049830', '269620', '222800',
         '010280', '115480', '900120', '123860', '149950', '099190', '040910', '068940', '090740', '078860', '114810',
         '031310', '124500', '260660', '096610', '061040', '310200', '102120', '065420', '103230', '101490', '095910',
         '096630', '043340', '096690', '021080', '950130', '070300', '092870', '048830', '058630', '019590', '097520',
         '265560', '045060', '048260', '080580', '049480', '109080', '019210', '082850', '074600', '014190', '008290',
         '095270', '065950', '036090', '203450', '241690', '078070', '054930', '336060', '367460', '331380', '178780',
         '179900', '047560', '900110', '123570', '353810', '090850', '134060', '079950', '064290', '254120', '234920',
         '174880', '040420', '079370', '094970', '023440', '080220', '082270', '034940', '036930', '119850', '065060',
         '071850', '342550', '042040', '054410', '274090', '078650', '126600', '352770', '139670', '079970', '105550',
         '026150', '246710', '131290', '033540', '037070', '065690', '225590', '005670', '203690', '335810', '032580',
         '024850', '341160', '221840', '222980', '050540', '078350', '123840', '007770', '102210', '034810', '048410',
         '039610', '115160', '054620', '035760', '058820', '065770', '083660', '367340', '037370', '036640', '030190',
         '024940', '036120', '048550', '317240', '038340', '192410', '217730', '063080', '044480', '223310']