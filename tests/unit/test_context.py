"""Follow-up detection, scope inheritance and rewrite guards (generation/context.py)."""
import pytest


def test_local_references_establish_memory_and_followups_inherit_scope():
    from manuora.generation.context import prepare_context, remember_reply
    docs = [{"product_id":"nextcloud", "product_version":"36", "document_id":"nc"},
            {"product_id":"kicad", "product_version":"10", "document_id":"kc"}]
    seed = prepare_context("In Nextcloud 36 I deleted ledger.csv and need to recover it.",
                           {}, docs, {}, thread_id="local")
    assert seed['clarification'] is None and not seed['needs_history']
    memory = remember_reply({}, seed, "Recovery instructions", [], turn_id="seed", kind="answer")
    follow = prepare_context("How do I recover it?", {}, docs, memory, thread_id="local")
    assert follow['needs_history'] and follow['scope']['product_version'] == '36'
    assert follow['history'][0]['turn_id'] == 'seed'
    switch = prepare_context("New topic: in KiCad 10, create a symbol and assign its footprint.",
                             {}, docs, memory, thread_id="local")
    assert switch['clarification'] is None and not switch['history']
    assert switch['scope']['product_id'] == 'kicad'
    assert memory['tasks'][0]['scope']['product_id'] == 'nextcloud'
    for question in ["How do I recover it?", "What does that button do?",
                     "In Nextcloud 36, how do I restore it?", "Make it shorter."]:
        assert prepare_context(question, {}, docs, {}, thread_id="empty")['clarification']
    assert prepare_context("In Nextcloud 35, restore the file and share it.",
                           {}, docs, memory, thread_id="local")['scope_conflict']


@pytest.mark.parametrize('question,needs_history', [
    ("Verify that synchronization is complete.", False),
    ("Confirming that synchronization is working.", False),
    ("After verifying that the files are synchronized, what next?", False),
    ("Can you confirm that icon?", True),
    ("Check that button is enabled.", True),
    ("Open the project and save it.", False),
    ("Connect the symbols and move them.", False),
    ("I see two files. How do I restore it?", True),
    ("Can I restore that file?", True),
    ("How about Linux?", True),
    ("Show me that.", True),
])
def test_reference_cues_distinguish_local_objects(question, needs_history):
    from manuora.generation.context import reference_signals
    assert reference_signals(question)['needs_prior_reference'] == needs_history


def test_context_scope_tasks_assistant_references_and_rewrite_guards(memory_model_transport, tmp_path):
    local, payloads, _ = memory_model_transport
    documents = [{'product_id':'nextcloud','product_version':v,'document_id':'nc'+v} for v in ('12','36')]
    documents += [{'product_id':'freecad','product_version':'0.16','document_id':'fc016'},
                  {'product_id':'freecad','product_version':'0.1','document_id':'fc01'}]
    path = str(tmp_path/'context.sqlite')
    with local.ManuoraGraphGenerator('fake','http://localhost/v1/chat/completions',db_path=path) as gen:
        def store(question, answer, thread='one', ui=None, sources=None):
            context = gen.prepare_retrieval_context(question,thread_id=thread,documents=documents,ui_scope=ui)
            gen.record_displayed_reply(question,answer,thread_id=thread,context=context,sources=sources)
            return context
        store('In Nextcloud 36 my older desktop client cannot use my account password after enabling 2FA.',
              'Create an app password [1].',ui={'document_id':'nc36'},
              sources=[{'citation':1,'chunk_id':'nc:1','document_id':'nc36','preview':'DO_NOT_STORE_BODY'}])
        follow=gen.prepare_retrieval_context('How do I configure that?',thread_id='one',documents=documents)
        assert follow['needs_history'] and follow['scope']['document_id']=='nc36'
        assert any(h['role']=='assistant' and 'app password' in h['content'] and h['sources'][0]['chunk_id']=='nc:1' for h in follow['history'])
        assert follow['scope_provenance']['document_id']['turn_id']
        corrected=gen.prepare_retrieval_context('Correction: I meant Nextcloud 12.',thread_id='one',documents=documents)
        assert corrected['scope']['product_version']=='12' and corrected['scope']['document_id'] is None
        assert 'older desktop client' in corrected['subject'] and '36' not in corrected['subject']
        assert gen.prepare_retrieval_context('Nextcloud 36 settings?',thread_id='one',documents=documents,ui_scope={'document_id':'nc12'})['clarification']
        assert not gen.prepare_retrieval_context('How do I configure that?',thread_id='other',documents=documents)['history']
        assert gen.prepare_retrieval_context('How do I configure that?',thread_id='other',documents=documents)['clarification']
        new=store('In Nextcloud 36, how do I customize dashboard widgets?', 'Open dashboard settings [1].')
        assert not new['history']  # Same product, different task.
        old=gen.prepare_retrieval_context('Back to Nextcloud: how do I connect the desktop client?',thread_id='one',documents=documents)
        assert any('app password' in h['content'] for h in old['history'])
        ambiguous=gen.prepare_retrieval_context('Back to Nextcloud: continue.',thread_id='one',documents=documents)
        assert ambiguous['clarification']
        store('In FreeCAD 0.16 I am using DXF import and export.', 'DXF is the exchange format [1].')
        dxf=gen.prepare_retrieval_context('Why is it limited to 2D?',thread_id='one',documents=documents)
        assert dxf['scope']['product_version']=='0.16' and dxf['scope']['document_id'] is None
        bad=gen.validate_rewrite(dxf,'Why is FreeCAD 0.16 limited to 2D?',documents=documents)
        assert not bad['accepted'] and 'dropped_identifier:DXF' in bad['issues']
        assert gen.validate_rewrite(dxf,'Why is DXF import and export in FreeCAD 0.16 limited to 2D?',documents=documents)['accepted']
        assert not gen.validate_rewrite(dxf,'Why is DXF import in FreeCAD 0.1 limited to 2D?',documents=documents)['accepted']
        defaults=gen.prepare_retrieval_context('How do I reset to defaults without deleting my files?',thread_id='one',documents=documents)
        checked=gen.validate_rewrite(defaults,'How do I reset in FreeCAD 0.16?',documents=documents)
        assert 'dropped_constraint:defaults' in checked['issues'] and 'dropped_constraint:negation' in checked['issues']
        state=gen.graph.get_state({'configurable':{'thread_id':'one'}}).values
        assert 'DO_NOT_STORE_BODY' not in str(state)
        assert not payloads  # Context preparation, storage and validation do not call a model.
    with local.ManuoraGraphGenerator('fake','http://localhost/v1/chat/completions',db_path=path) as reopened:
        assert reopened.prepare_retrieval_context('And on Linux?',thread_id='one',documents=documents)['needs_history']
