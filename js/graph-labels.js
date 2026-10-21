/* 图谱展示用中文名称；检索和缓存继续使用原有关系编码。 */
globalThis.GraphLabels = (() => {
  const relations = Object.freeze({
    APPEARS_IN: '登场于', PARENT_OF: '父母', GRANDPARENT_OF: '祖父母',
    SPOUSE_OF: '配偶', COUSIN_OF: '表亲', UNCLE_OF: '叔伯',
    DESCENDANT_OF: '后代', LIKES: '喜欢', FRIEND_OF: '好朋友',
    PARTNER_OF: '搭档', SEALED_BY: '被封印', RELEASED_BY: '被解封',
    HEAD_OF: '负责人', LIVES_IN: '居住于', STUDENT_OF: '就读于',
    LOCATED_IN: '位于', MEMBER_OF: '成员', LEADER_OF: '首领',
    WISHED_ON: '向其许愿', ALIAS: '别名', UNIT_NO: '编号',
    PROTOTYPE: '原型', ROLE: '身份', BIRTHDAY: '生日',
    CATCHPHRASE: '口头禅', WEAPON: '武器', SPECIAL_MOVE: '必杀技',
    FAVORITE_FOOD: '最爱食物', CARRIES: '随身物品', TRANSFORM_TIME: '变身时限',
    GIANT_FORM: '巨人形态', FACTION: '阵营', FIRST_AIRED: '首播时间',
    BROADCASTER: '播出平台', EPISODES: '集数', THEME_SONG: '主题曲',
    FIRST_MOVIE: '首部电影', QUANTITY: '数量',
  });
  const types = Object.freeze({
    Show: '节目', Character: '角色', Place: '地点', Group: '团体',
    Item: '道具', Literal: '属性值', Unknown: '未知',
  });
  const intents = Object.freeze({
    relation_path: '人物间的关系路径', relation_chain: '指定关系链',
    neighborhood: '周边关系', 'neighborhood(fallback)': '周边关系（备用）',
  });
  const routes = Object.freeze({graph: '图检索', lexical: '文本检索'});
  const label = (map, key) => Object.hasOwn(map, key) ? map[key] : (key || '');
  return Object.freeze({
    relations,
    relation: key => label(relations, key),
    type: key => label(types, key),
    intent: key => label(intents, key),
    route: key => label(routes, key),
    edgeTitle(edge, nodes) {
      if (!edge) return '';
      const source = edge.reversed ? edge.to : edge.from;
      const target = edge.reversed ? edge.from : edge.to;
      return `${nodes[source]?.name || source} —${label(relations, edge.rel)}→ ${nodes[target]?.name || target}`;
    },
  });
})();
