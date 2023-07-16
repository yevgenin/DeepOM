import re
from typing import TextIO

from Bio import SeqIO
from Bio.Seq import reverse_complement
from Bio.SeqIO.FastaIO import FastaIterator
from devtools import debug
from pydantic import BaseModel

from utils.pyutils import NDArray


class GenomePatternMapper:
    class Config(BaseModel):
        pattern: str

        # for human genome, gives the chromosomes
        record_id_prefix = 'NC_'

    class Input(BaseModel):
        seq: str
        id: str

    class Output(BaseModel):
        positions: NDArray
        id: str

    def __init__(self, **config):
        self.config = self.Config(**config)
        debug(self, self.config)
        self.regex_pattern = self._regex_pattern()

    def find_pattern(self, sequence_record: dict):
        """
        does case-insensitive search for pattern in sequence
        """
        # noinspection PyTypeChecker
        _sequence_record = self.Input(**sequence_record)
        return self.Output(
            positions=[m.start() for m in re.finditer(self.regex_pattern, _sequence_record.seq, re.I)],
            id=_sequence_record.id,
        ).dict()

    def sequence_records(self, fasta_file: str | TextIO):
        """
        filters records by id prefix
        """
        record: SeqIO.SeqRecord
        yield from (
            self.Input(seq=str(record.seq), id=record.id).dict()
            for record in FastaIterator(fasta_file)
            if record.id.startswith(self.config.record_id_prefix)
        )

    def _regex_pattern(self):
        rev_complement = reverse_complement(self.config.pattern)
        if self.config.pattern == rev_complement:
            return self.config.pattern
        else:
            return f'{self.config.pattern}|{rev_complement}'
